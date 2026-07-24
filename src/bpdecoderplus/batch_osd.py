import torch
import numpy as np
from typing import List, Tuple, Optional

class BatchOSDDecoder:
    """GPU-accelerated OSD decoder for batch syndrome processing."""

    def __init__(self, H: np.ndarray, device: str = 'cuda'):
        """
        Initialize GPU-accelerated OSD decoder.

        Args:
            H: Parity Check Matrix (m rows, n columns)
            device: torch device ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.H = H.astype(np.int8)
        self.num_checks, self.num_errors = H.shape

        # Cache for RREF computation (kept on CPU)
        self._cached_rref = None
        self._cached_pivot_cols = None
        self._cached_column_order = None

    def _compute_soft_weight_gpu(self, solutions: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute soft-weighted cost using standard log-probability weight (GPU version).

        Args:
            solutions: (num_candidates, n) - binary error patterns
            probs: (n,) - error probabilities from BP

        Returns:
            costs: (num_candidates,) - soft-weighted costs (lower is better)
        """
        probs_clipped = torch.clamp(probs, 1e-10, 1 - 1e-10)
        log_weights = -torch.log(probs_clipped)
        costs = (solutions * log_weights).sum(dim=1)
        return costs

    def _generate_osd_cs_candidates(self, k: int, osd_order: int) -> np.ndarray:
        """
        Generate OSD-CS (Combination Sweep) candidate strings.

        Args:
            k: Number of free variables
            osd_order: Maximum number of positions to consider for flips

        Returns:
            Array of candidate bit patterns (num_candidates, k)
        """
        candidates = []

        # Zero vector (all free variables = 0)
        candidates.append(np.zeros(k, dtype=np.int8))

        # Single-bit flips
        for i in range(k):
            candidate = np.zeros(k, dtype=np.int8)
            candidate[i] = 1
            candidates.append(candidate)

        # Two-bit flips (within osd_order)
        for i in range(min(osd_order, k)):
            for j in range(i + 1, min(osd_order, k)):
                candidate = np.zeros(k, dtype=np.int8)
                candidate[i] = 1
                candidate[j] = 1
                candidates.append(candidate)

        return np.array(candidates, dtype=np.int8)

    def _evaluate_candidates_gpu(self, candidates: torch.Tensor, augmented: np.ndarray,
                                  search_cols: List[int], probs_sorted: torch.Tensor,
                                  pivot_cols: List[int]) -> torch.Tensor:
        """
        Evaluate all OSD candidates in parallel on GPU.

        Args:
            candidates: (num_candidates, k) - candidate patterns for free variables
            augmented: RREF augmented matrix [H | s] (on CPU)
            search_cols: Column indices being searched
            probs_sorted: (n,) - sorted error probabilities (on GPU)
            pivot_cols: List of pivot column indices

        Returns:
            best_solution: (n,) - best solution in sorted order
        """
        num_candidates = candidates.shape[0]

        # Transfer necessary data to GPU
        M_subset = torch.from_numpy(augmented[:, search_cols]).float().to(self.device)
        syndrome_col = torch.from_numpy(augmented[:, -1]).float().to(self.device)

        # Convert candidates to float for matrix multiplication
        candidates_float = candidates.float()

        # Compute target syndromes for all candidates in parallel
        # target_syndrome = (s + M @ e_T) % 2
        target_syndromes = (syndrome_col.unsqueeze(0) + candidates_float @ M_subset.T) % 2  # (num_candidates, m)

        # Initialize all candidate solutions
        cand_solutions = torch.zeros(num_candidates, self.num_errors, device=self.device)

        # Set search columns (free variables)
        search_cols_tensor = torch.tensor(search_cols, device=self.device)
        cand_solutions[:, search_cols_tensor] = candidates_float

        # Set pivot variables based on target syndromes (vectorized)
        # Build pivot row mapping
        augmented_torch = torch.from_numpy(augmented[:, :self.num_errors]).to(self.device)
        pivot_cols_set = set(pivot_cols)
        for r in range(augmented.shape[0]):
            # Find pivot column in this row
            row_pivots = torch.where(augmented_torch[r, :] == 1)[0]
            if len(row_pivots) > 0:
                pivot_c = row_pivots[0].item()
                if pivot_c in pivot_cols_set:
                    # Set this pivot variable for all candidates
                    cand_solutions[:, pivot_c] = target_syndromes[:, r]

        # Compute costs for all candidates
        costs = self._compute_soft_weight_gpu(cand_solutions, probs_sorted)

        # Return best solution
        best_idx = torch.argmin(costs)
        return cand_solutions[best_idx]

    def solve(self, syndrome: np.ndarray, error_probs: np.ndarray,
              osd_order: int = 10, osd_method: str = 'exhaustive',
              random_seed: Optional[int] = None) -> np.ndarray:
        """
        Solve for the most likely error pattern using GPU-accelerated OSD.

        Args:
            syndrome: the observed syndrome (binary array of length m)
            error_probs: error probabilities from BP for each variable (array of length n)
            osd_order: the search depth (lambda)
            osd_method: 'exhaustive' (default) or 'combination_sweep' (OSD-CS)
            random_seed: optional random seed for deterministic behavior

        Returns:
            Estimated error pattern (binary array of length n)
        """
        # 1. Convert probabilities to numpy array and clip to valid range
        probs = np.asarray(error_probs, dtype=np.float64).flatten()
        if len(probs) != self.num_errors:
            raise ValueError(f"error_probs length {len(probs)} doesn't match number of errors {self.num_errors}")

        # Clip probabilities to avoid numerical issues
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        # 2. Sort (Soft Decision)
        # For quantum error correction with low error rates, sort by probability descending.
        # This ensures high-probability errors (those BP identified as likely errors)
        # are placed first and become pivots in RREF. Low-probability errors become
        # free variables and are set to 0 in OSD-0.
        #
        # Note: Traditional OSD uses |p - 0.5| (reliability), but this doesn't work
        # well for quantum codes where:
        # - Most errors have p ≈ 0 → reliability ≈ 0.5
        # - Identified errors have p ≈ 1 → reliability ≈ 0.5
        # Both have similar reliability despite being very different!
        # Sorting by probability directly is more appropriate for this use case.
        sorted_indices = np.argsort(probs)[::-1]  # Highest probability first

        # 3. Build the augmented matrix [H_sorted | s] and compute RREF (on CPU)
        augmented, pivot_cols = self._get_rref_cached(sorted_indices, syndrome)

        # Determine the column indices of the free variables (in the sorted space)
        all_cols = set(range(self.num_errors))
        free_cols = sorted(list(all_cols - set(pivot_cols)))

        # 5. OSD post-processing (Searching)
        # Basis solution (OSD-0 Solution): Assume all free variables are 0
        solution_base = np.zeros(self.num_errors, dtype=np.int8)

        # Build the row mapping: the pivot column c corresponds to which row r
        pivot_row_map = {}
        pivot_cols_set = set(pivot_cols)
        for r in range(augmented.shape[0]):
            row_pivots = np.where(augmented[r, :self.num_errors] == 1)[0]
            if len(row_pivots) > 0:
                col = row_pivots[0]
                if col in pivot_cols_set:
                    pivot_row_map[col] = r
                    solution_base[col] = augmented[r, -1]

        # If osd_order == 0, return the OSD-0 result directly
        if osd_order == 0:
            final_solution_sorted = solution_base
        else:
            # --- GPU-Accelerated OSD-E ---
            # Search over free variables with highest probability (most suspicious).
            # Get probabilities of free variables in sorted order
            free_cols_with_prob = [(col, probs[sorted_indices[col]]) for col in free_cols]
            # Sort by probability descending (highest first - most suspicious)
            free_cols_with_prob.sort(key=lambda x: -x[1])
            # Select top osd_order free variables
            search_cols = [col for col, _ in free_cols_with_prob[:osd_order]]

            # Transfer sorted probabilities to GPU
            probs_sorted = torch.from_numpy(probs[sorted_indices]).float().to(self.device)

            # Generate candidates based on method
            if osd_method == 'combination_sweep':
                candidates_np = self._generate_osd_cs_candidates(len(search_cols), osd_order)
            else:
                # Exhaustive: Generate all 2^k combinations
                num_candidates = 1 << len(search_cols)
                candidates_np = np.array([[(i >> j) & 1 for j in range(len(search_cols))]
                                         for i in range(num_candidates)], dtype=np.int8)

            # Transfer candidates to GPU
            candidates = torch.from_numpy(candidates_np).to(self.device)

            # Evaluate all candidates in parallel on GPU
            best_solution_sorted = self._evaluate_candidates_gpu(
                candidates, augmented, search_cols, probs_sorted, pivot_cols
            )

            # Convert back to numpy
            final_solution_sorted = best_solution_sorted.cpu().numpy().astype(int)

        # 6. Inverse mapping (Unsort)
        estimated_errors = np.zeros(self.num_errors, dtype=int)
        estimated_errors[sorted_indices] = final_solution_sorted

        return estimated_errors

    def solve_batch(self, syndromes: np.ndarray, error_probs: np.ndarray,
                    osd_order: int = 10, osd_method: str = 'exhaustive') -> np.ndarray:
        """
        Solve multiple syndromes in batch (currently sequential, can be optimized).

        Args:
            syndromes: (batch_size, m) - array of syndromes
            error_probs: (batch_size, n) - error probabilities for each syndrome
            osd_order: the search depth
            osd_method: 'exhaustive' or 'combination_sweep'

        Returns:
            solutions: (batch_size, n) - estimated error patterns
        """
        batch_size = syndromes.shape[0]
        solutions = np.zeros((batch_size, self.num_errors), dtype=int)

        for i in range(batch_size):
            solutions[i] = self.solve(syndromes[i], error_probs[i], osd_order, osd_method)

        return solutions

    def _get_rref_cached(self, sorted_indices: np.ndarray, syndrome: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Get RREF with caching. Cache is valid when column order is unchanged.

        Args:
            sorted_indices: Column permutation order
            syndrome: Syndrome vector

        Returns:
            Tuple of (augmented matrix in RREF, pivot columns)
        """
        # IMPORTANT: Caching disabled due to bug where syndrome column wasn't being
        # transformed by the same row operations as the cached RREF matrix.
        # This caused invalid solutions that didn't satisfy the syndrome.
        # TODO: Implement proper caching by storing row operations and applying them to new syndromes

        # Always compute fresh RREF
        H_sorted = self.H[:, sorted_indices]
        augmented = np.hstack([H_sorted, syndrome.reshape(-1, 1)]).astype(np.int8)
        pivot_cols = self._compute_rref(augmented)

        return augmented, pivot_cols

    def _compute_rref(self, M: np.ndarray) -> List[int]:
        """
        Compute the reduced row echelon form (RREF) of matrix M - In-place modification
        """
        m, n = M.shape
        num_cols_to_scan = n - 1

        pivot_row = 0
        pivot_cols = []

        for col in range(num_cols_to_scan):
            if pivot_row >= m:
                break

            candidates = np.where(M[pivot_row:, col] == 1)[0]

            if len(candidates) == 0:
                continue

            swap_r = candidates[0] + pivot_row
            if swap_r != pivot_row:
                M[[pivot_row, swap_r]] = M[[swap_r, pivot_row]]

            pivot_cols.append(col)

            rows_to_xor = np.where(M[:, col] == 1)[0]
            rows_to_xor = rows_to_xor[rows_to_xor != pivot_row]

            if len(rows_to_xor) > 0:
                M[rows_to_xor, :] ^= M[pivot_row, :]

            pivot_row += 1

        return pivot_cols
