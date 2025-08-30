"""
geometric_gpe.py
=================

This module implements a simplified version of the Geometric Persistent Excitation (G‑PE)
criterion described in the accompanying report.  It provides tools for evaluating
whether a collection of short trajectory segments, together with their control inputs,
satisfy the geometric assumptions required to guarantee good conditioning of data‐driven
Koopman/EDMD models.  The implementation focuses on proxy metrics that are practical
to compute from data:

* **Direction Coverage** – ensure that the set of displacement vectors spans the
  controllable subspace with a prescribed angular resolution.
* **Non‑Clustering** – verify that trajectory segments are not overly concentrated in
  narrow tubes across multiple length scales.  We approximate the Katz–Tao/Frostman
  Wolff axioms by counting the number of segment centres contained in sliding boxes of
  a given size and comparing to the bound predicted by the Kakeya theory.
* **Covariance Conditioning** – monitor the smallest eigenvalue of the state and
  lifted‑state (feature) covariance matrices, which are proxies for the minimum
  singular value of the EDMD regression matrix.  A positive lower bound implies
  stable least‑squares estimates.

The code is intended as an educational reference rather than a highly optimised
implementation.  Researchers are encouraged to adapt the parameters and check
functions based on the specific geometry of their system and the dimensionality of
the controllable subspace.  See the report for further discussion and limitations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Tuple


@dataclass
class GPEParameters:
    """Parameters specifying the desired geometric resolution and non‑clustering.

    Attributes
    ----------
    delta : float
        Resolution on the unit sphere.  Directions whose angular separation is less
        than ``delta`` are considered the same direction for coverage counting.
    M_expected : int
        Expected minimum number of distinct directions.  For an n_c‑dimensional
        controllable subspace, a rough guideline is ``M_expected ~ c * delta^{-(n_c-1)}``.
    rho_max : float
        Maximum acceptable value for the non‑clustering ratio.  Values less than
        one indicate that no narrow slab contains an unexpectedly large number of
        segments.
    scales : Iterable[float]
        Sequence of scales ``r`` at which to evaluate the non‑clustering count.
        Typical values span from ``delta`` up to 1 (in units of trajectory length).
    slab_length : float
        Half‑length of the sliding box along the direction axis.  This value
        should be comparable to the average segment length.  Increasing this
        length reduces sensitivity to local fluctuations.
    """

    delta: float
    M_expected: int
    rho_max: float = 1.0
    scales: Iterable[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    slab_length: float = 1.0


def compute_directions(X: np.ndarray) -> np.ndarray:
    """Compute unit direction vectors for each consecutive pair in the sequence.

    Parameters
    ----------
    X : array, shape (N, n)
        Sequence of state vectors.

    Returns
    -------
    dirs : array, shape (N-1, n)
        Normalised difference vectors ``(x_{k+1} - x_k) / ||x_{k+1} - x_k||``.

    Notes
    -----
    Zero‑length displacements are ignored (they produce NaNs); such rows
    are removed from the result.
    """
    diffs = np.diff(X, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    mask = norms > 0
    if mask.sum() == 0:
        return np.empty((0, X.shape[1]))
    dirs = (diffs[mask].T / norms[mask]).T
    return dirs


def count_distinct_directions(dirs: np.ndarray, delta: float) -> int:
    """Count the number of directions that are pairwise separated by at least ``delta``.

    A greedy algorithm is used: choose a direction, remove all directions within
    angular radius ``delta``, and repeat.

    Parameters
    ----------
    dirs : array, shape (M, n)
        Unit direction vectors on the unit sphere.
    delta : float
        Angular resolution (in radians).  Must be small (0 < delta < π).

    Returns
    -------
    int
        Number of representative directions whose pairwise angular distance
        exceeds ``delta``.
    """
    if len(dirs) == 0:
        return 0
    remaining = dirs.copy()
    selected_count = 0
    while len(remaining) > 0:
        chosen = remaining[0]
        selected_count += 1
        # Cosine similarity threshold for separation.
        cos_thresh = np.cos(delta)
        dots = remaining @ chosen
        remaining = remaining[dots < cos_thresh]
    return selected_count


def approximate_non_clustering(
    X: np.ndarray,
    dirs: np.ndarray,
    scales: Iterable[float],
    slab_length: float,
    n_dims: int,
) -> float:
    """Compute a proxy for the non‑clustering ratio across multiple scales.

    For each scale ``r`` in ``scales`` and for each representative direction in ``dirs``,
    this function counts the number of segment midpoints that lie within a narrow
    rectangular box aligned with that direction: along the direction, the box has
    half‑length ``slab_length``; orthogonally, it has radius ``r``.  The maximum
    count over all boxes is normalised by ``M * r^(n_dims-1)``.  Returning a value
    close to 1 indicates adherence to the Wolff non‑clustering bound; larger
    values suggest clustering.

    Parameters
    ----------
    X : array, shape (N, n)
        State trajectory.
    dirs : array, shape (M, n)
        Selected unit direction vectors (from ``compute_directions``).  These
        define the orientation of the sliding boxes.
    scales : iterable of float
        Radii at which to evaluate clustering.  Typically spans ``delta`` to 1.
    slab_length : float
        Half‑length of the box along the direction axis.  Must be positive.
    n_dims : int
        Dimension of the controllable subspace (2 or 3).  Exponent in the
        normalising factor ``r^(n_dims-1)``.

    Returns
    -------
    float
        The maximum normalised count over all scales and directions.

    Notes
    -----
    - This is a heuristic proxy; it does not perfectly replicate the Katz–Tao or
      Frostman Wolff axioms, but captures their spirit: if many segments lie
      within a thin tube aligned along one direction, clustering is detected.
    - Complexity scales with O(M * scales * N).  For large datasets, consider
      subsampling or random selection of directions.
    """
    if len(dirs) == 0:
        return 0.0
    # Compute midpoints of segments.
    mids = 0.5 * (X[1:] + X[:-1])
    max_ratio = 0.0
    for r in scales:
        # Precompute r^(n_dims-1) for normalisation.
        denom = (r ** (n_dims - 1)) * len(dirs)
        # Consider a subset of directions to reduce cost (randomly pick K).
        for d in dirs:
            # orthonormal basis: direction d and orthogonal complement
            d_norm = d / (np.linalg.norm(d) + 1e-12)
            # compute coordinates along d and perpendicular distance
            proj = mids @ d_norm
            resid = mids - np.outer(proj, d_norm)
            perp_norm = np.linalg.norm(resid, axis=1)
            # For boxes: count points where perp_norm <= r and |proj - c| <= slab_length
            # We slide along the projection axis: build a histogram of counts in windows
            # of width 2*slab_length.
            if len(proj) == 0:
                continue
            # Sort by projected coordinate.
            idx = np.argsort(proj)
            sorted_proj = proj[idx]
            sorted_perp = perp_norm[idx]
            # Two pointer sliding window.
            left = 0
            right = 0
            max_count_dir = 0
            while left < len(sorted_proj):
                # Move right pointer while within box.
                while right < len(sorted_proj) and (sorted_proj[right] - sorted_proj[left] <= 2 * slab_length):
                    right += 1
                # Count points with perp distance <= r in current window.
                window_perp = sorted_perp[left:right]
                count = int((window_perp <= r).sum())
                if count > max_count_dir:
                    max_count_dir = count
                left += 1
            # Normalise by expected count and update max ratio.
            ratio = max_count_dir / denom if denom > 0 else 0.0
            if ratio > max_ratio:
                max_ratio = ratio
    return max_ratio


def covariance_min_eig(X: np.ndarray) -> float:
    """Compute the smallest eigenvalue of the sample covariance of X.

    Parameters
    ----------
    X : array, shape (N, n)
        Observations; rows are samples.

    Returns
    -------
    float
        The minimum eigenvalue of the zero‑mean covariance matrix.  If fewer than
        two samples are provided, returns 0.
    """
    if X.shape[0] < 2:
        return 0.0
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / float(Xc.shape[0])
    vals = np.linalg.eigvalsh(C)
    return float(np.min(vals))


def evaluate_gpe(
    X: np.ndarray,
    psi_func: Callable[[np.ndarray], np.ndarray],
    params: GPEParameters,
    n_dims: int,
) -> Tuple[float, float, int, float, float, float]:
    """Evaluate the G‑PE criterion on a trajectory.

    Parameters
    ----------
    X : array, shape (N, n)
        State trajectory (rows are samples).  Only the first ``n_dims`` coordinates
        are used for geometric calculations.
    psi_func : function
        Lifting/feature map ``psi`` used for Koopman/EDMD.  Should return a 1D
        array of features for a 1D state input.
    params : GPEParameters
        Settings for resolution, expected number of directions, etc.
    n_dims : int
        Dimension of the controllable subspace used for geometric checks
        (typically 2 or 3).

    Returns
    -------
    gpe_index : float
        Minimum of the normalised scores (see discussion below); a value ≥ 1
        indicates that all thresholds are met.
    ratio_dirs : float
        Ratio of distinct directions to expected directions ``M_expected``.
    distinct_dirs : int
        Actual count of distinct directions.
    ratio_non_clust : float
        Normalised non‑clustering ratio; values ≤ 1 are desirable.
    lambda_min_state : float
        Minimum eigenvalue of state covariance ``Σ_x``.
    lambda_min_lift : float
        Minimum eigenvalue of lifted state covariance ``Σ_ψ`` (using
        demeaned features).

    Notes
    -----
    - This function focuses on the geometric aspects of G‑PE; it does not
      estimate \\lambda (density) since in typical control experiments one uses
      the full segment data (\lambda=1).  If shorter subsegments are used for
      shading, modify ``evaluate_gpe`` accordingly to weight segments.
    - To monitor smallest singular values of the EDMD regression matrix, one
      should construct the Hankel/feature matrix and compute its spectrum or
      apply incremental algorithms on the fly.
    """
    # Extract subspace coordinates for geometric checks.
    X_sub = X[:, :n_dims]
    # Compute directions and distinct count.
    dirs_full = compute_directions(X_sub)
    distinct_dirs = count_distinct_directions(dirs_full, params.delta)
    ratio_dirs = distinct_dirs / float(params.M_expected) if params.M_expected > 0 else 0.0
    # Non‑clustering ratio (proxy).
    # Use a reduced set of directions for efficiency: e.g., the first K directions.
    # If too few directions, skip clustering check.
    K = min(20, len(dirs_full))  # limit to 20 directions for efficiency
    if K > 0:
        dirs_sample = dirs_full[:K]
        ratio_non_clust = approximate_non_clustering(
            X_sub, dirs_sample, params.scales, params.slab_length, n_dims
        )
    else:
        ratio_non_clust = 0.0
    # Covariance spectral quantities.
    lambda_min_state = covariance_min_eig(X_sub)
    # Compute lifted state covariance (excluding constant feature if present).
    if psi_func is not None:
        Phi = np.stack([psi_func(x) for x in X], axis=0)
        # Remove constant feature if it exists (last element of phi = 1.0).
        # Find columns that vary.
        variances = Phi.var(axis=0)
        non_zero_cols = variances > 1e-12
        if np.sum(non_zero_cols) > 0:
            Phi_use = Phi[:, non_zero_cols]
            lambda_min_lift = covariance_min_eig(Phi_use)
        else:
            lambda_min_lift = 0.0
    else:
        lambda_min_lift = 0.0
    # Compute a single GPE‑index: take the minimum of normalised scores.
    s1 = ratio_dirs  # ~1 means enough directions
    s2 = (params.rho_max / ratio_non_clust) if ratio_non_clust > 0 else np.inf
    s3 = lambda_min_state / 1.0  # normalise by a baseline (here 1.0) – user‑defined
    s4 = lambda_min_lift / 1.0   # likewise
    # Drop infinite values if ratio_non_clust=0 (too few segments).
    scores = [s1, s2, s3, s4]
    # Avoid NaNs
    scores = [s for s in scores if np.isfinite(s)]
    gpe_index = min(scores) if len(scores) > 0 else 0.0
    return (
        gpe_index,
        ratio_dirs,
        distinct_dirs,
        ratio_non_clust,
        lambda_min_state,
        lambda_min_lift,
    )


def example_usage():
    """Example showing how to compute G‑PE metrics on a synthetic trajectory.

    This function simulates a random walk in two dimensions and evaluates the
    geometric coverage and covariance conditioning metrics.  It is provided
    solely for demonstration; remove or modify in production code.
    """
    # Generate a random walk in 2D
    np.random.seed(42)
    N = 200
    X = np.cumsum(np.random.randn(N, 2), axis=0)
    # Lifting: 3rd order polynomial features (without constant term)
    def poly3(z):
        x1, x2 = z
        return np.array([
            x1, x2, x1**2, x1 * x2, x2**2, x1**3, x1**2 * x2, x1 * x2**2, x2**3
        ])
    # Define GPE parameters
    params = GPEParameters(delta=0.3, M_expected=10, rho_max=1.5, scales=[0.5, 1.0], slab_length=1.0)
    # Evaluate
    metrics = evaluate_gpe(X, poly3, params, n_dims=2)
    print("GPE index, ratio_dirs, distinct_dirs, ratio_non_clust, lambda_min_state, lambda_min_lift:")
    print(metrics)


if __name__ == "__main__":
    example_usage()