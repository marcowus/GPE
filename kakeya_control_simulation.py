import numpy as np


def direction_codebook(n_dirs):
    """Return n_dirs unit vectors uniformly spread on the unit circle."""
    angles = np.linspace(0, np.pi, n_dirs, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def sample_start_points(n_segments, clustered=False, box_size=0.1):
    """Sample start points in unit square.

    If ``clustered`` is True, all points are sampled inside a small
    square of side length ``box_size`` located at the origin. Otherwise
    they are sampled uniformly from the unit square.
    """
    if clustered:
        return np.random.rand(n_segments, 2) * box_size
    return np.random.rand(n_segments, 2)


def generate_segments(directions, start_points, length=0.2):
    """Generate line segments as (start, end) pairs."""
    segments = []
    for s, d in zip(start_points, directions):
        segments.append((s, s + length * d))
    return segments


def union_area(segments, width=0.02, grid_size=400):
    """Approximate the area of the union of thickened segments.

    The approximation uses a regular grid over the unit square and marks
    points that fall within ``width / 2`` distance of any segment.
    """
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel()], axis=1)
    covered = np.zeros(len(points), dtype=bool)
    half_width = width / 2.0
    for s, e in segments:
        d = e - s
        length = np.linalg.norm(d)
        if length == 0:
            continue
        d /= length
        rel = points - s
        t = np.clip(rel @ d, 0, length)
        dist = np.linalg.norm(rel - np.outer(t, d), axis=1)
        covered |= dist <= half_width
    area = covered.mean()
    return area


def simulate(n_dirs=20, clustered=False, seed=0):
    np.random.seed(seed)
    dirs = direction_codebook(n_dirs)
    starts = sample_start_points(n_dirs, clustered=clustered)
    segs = generate_segments(dirs, starts)
    # lambda_min of covariance of start points
    cov = np.cov(starts.T)
    lam_min = np.linalg.eigvalsh(cov)[0]
    area = union_area(segs)
    return lam_min, area


if __name__ == "__main__":
    baseline = simulate(clustered=False)
    clustered = simulate(clustered=True)
    print("Baseline λ_min: {:.5f}, union area: {:.5f}".format(*baseline))
    print("Clustered λ_min: {:.5f}, union area: {:.5f}".format(*clustered))
    print("λ_min ratio (clustered / baseline): {:.3f}".format(
        clustered[0] / baseline[0]))
    print("Area ratio (clustered / baseline): {:.3f}".format(
        clustered[1] / baseline[1]))
