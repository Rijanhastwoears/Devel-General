import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def match_ground_truth(
    alt_data: pl.DataFrame,
    ground_truth: pl.DataFrame,
    *,
    tolerance_mz: float | None = None,
    tolerance_rt: float | None = None,
) -> pl.DataFrame:
    """
    Match alternative mass spec rows to ground-truth rows using Polars asof joins.

    For each row in alt_data this finds:
      - nearest ground-truth row by mz (adds mz_by_mz, rt_by_mz)
      - nearest ground-truth row by rt (adds mz_by_rt, rt_by_rt)

    Optional tolerances (absolute) can be passed; if the nearest neighbour is
    further than the tolerance the matched columns will be null.

    Notes
    - Inputs are not mutated; a new DataFrame is returned.
    - Polars join_asof requires inputs sorted by the join key. To preserve the
      original ordering we add a row index, perform the asof joins on sorted
      copies, then reattach matched columns by index.
    """
    # Basic checks
    if "mz" not in alt_data.columns or "rt" not in alt_data.columns:
        raise ValueError("alt_data must contain 'mz' and 'rt' columns")
    if "mz" not in ground_truth.columns or "rt" not in ground_truth.columns:
        raise ValueError("ground_truth must contain 'mz' and 'rt' columns")

    # Coerce to floats for safe numeric comparisons
    alt = alt_data.with_columns(pl.col("mz").cast(pl.Float64), pl.col("rt").cast(pl.Float64))
    gt = ground_truth.with_columns(pl.col("mz").cast(pl.Float64), pl.col("rt").cast(pl.Float64))

    # Prepare ground-truth frames sorted for asof joins; alias columns to produce clear result names
    gt_by_mz = gt.sort("mz").select([pl.col("mz").alias("mz_by_mz"), pl.col("rt").alias("rt_by_mz"), pl.col("mz")])
    gt_by_rt = gt.sort("rt").select([pl.col("mz").alias("mz_by_rt"), pl.col("rt").alias("rt_by_rt"), pl.col("rt")])

    # Add index to restore original order later
    alt_indexed = alt.with_row_count("_alt_idx")

    # As-of join by mz (sort by mz on the left for join_asof)
    joined_mz_idx = (
        alt_indexed.sort("mz")
        .join_asof(
            gt_by_mz,
            left_on="mz",
            right_on="mz",
            strategy="nearest",
            tolerance=tolerance_mz,
        )
        .select("_alt_idx", "mz_by_mz", "rt_by_mz")
    )

    # As-of join by rt (sort by rt on the left for join_asof)
    joined_rt_idx = (
        alt_indexed.sort("rt")
        .join_asof(
            gt_by_rt,
            left_on="rt",
            right_on="rt",
            strategy="nearest",
            tolerance=tolerance_rt,
        )
        .select("_alt_idx", "mz_by_rt", "rt_by_rt")
    )

    # Reattach matched columns to original order using the index
    result = (
        alt_indexed.join(joined_mz_idx, on="_alt_idx", how="left")
                   .join(joined_rt_idx, on="_alt_idx", how="left")
                   .sort("_alt_idx")
                   .drop("_alt_idx")
    )

    return result


def calculate_differences(matched_data: pl.DataFrame) -> pl.DataFrame:
    """
    Given the output of match_ground_truth, compute ppm and RT differences.

    Produces:
      - ppm_diff_by_mz
      - dalton_diff_by_mz
      - dalton_diff_by_rt
      - rt_diff_by_mz
      - ppm_diff_by_rt
      - rt_diff_by_rt
    """
    result = matched_data.with_columns(
        (abs(pl.col("mz") - pl.col("mz_by_mz"))).alias("dalton_diff_by_mz"),
        (abs(pl.col("mz") - pl.col("mz_by_rt"))).alias("dalton_diff_by_rt"),
        ((abs(pl.col("mz") - pl.col("mz_by_mz")) / pl.col("mz_by_mz")) * 1_000_000).alias("ppm_diff_by_mz"),
        (pl.col("rt") - pl.col("rt_by_mz")).abs().alias("rt_diff_by_mz"),
        ((abs(pl.col("mz") - pl.col("mz_by_rt")) / pl.col("mz_by_rt")) * 1_000_000).alias("ppm_diff_by_rt"),
        (pl.col("rt") - pl.col("rt_by_rt")).abs().alias("rt_diff_by_rt"),
    )
    return result


def plot_tolerance_scatter(x_series: pl.Series, y_series: pl.Series,
                           x_tol: float, y_tol: float,
                           x_label: str = None, y_label: str = None,
                           title: str = None, figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Create an academic-style scatter plot with tolerance-based color coding.

    Color scheme:
      - Green: Below both x_tol and y_tol (good matches)
      - Blue: Below x_tol only
      - Yellow: Below y_tol only
      - Red: Above both tolerances (poor matches)
    """
    x_data = x_series.to_numpy()
    y_data = y_series.to_numpy()

    below_x = np.abs(x_data) < x_tol
    below_y = np.abs(y_data) < y_tol

    green_mask = below_x & below_y
    blue_mask = below_x & ~below_y
    yellow_mask = ~below_x & below_y
    red_mask = ~below_x & ~below_y

    fig, ax = plt.subplots(figsize=figsize)

    if np.any(green_mask):
        ax.scatter(x_data[green_mask], y_data[green_mask],
                   c='green', alpha=0.6, s=50, label='Within both tolerances',
                   edgecolors='darkgreen', linewidth=0.5)

    if np.any(blue_mask):
        ax.scatter(x_data[blue_mask], y_data[blue_mask],
                   c='blue', alpha=0.6, s=50, label='Within x-tolerance only',
                   edgecolors='darkblue', linewidth=0.5)

    if np.any(yellow_mask):
        ax.scatter(x_data[yellow_mask], y_data[yellow_mask],
                   c='gold', alpha=0.6, s=50, label='Within y-tolerance only',
                   edgecolors='orange', linewidth=0.5)

    if np.any(red_mask):
        ax.scatter(x_data[red_mask], y_data[red_mask],
                   c='red', alpha=0.6, s=50, label='Outside both tolerances',
                   edgecolors='darkred', linewidth=0.5)

    ax.axvline(x=x_tol, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'x-tolerance = ±{x_tol}')
    ax.axvline(x=-x_tol, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=y_tol, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'y-tolerance = ±{y_tol}')
    ax.axhline(y=-y_tol, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.axvspan(-x_tol, x_tol, alpha=0.1, color='green')
    ax.axhspan(-y_tol, y_tol, alpha=0.1, color='green')

    ax.set_xlabel(x_label or x_series.name or 'X', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label or y_series.name or 'Y', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Tolerance Analysis', fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)

    total_points = len(x_data)
    green_count = np.sum(green_mask)
    blue_count = np.sum(blue_mask)
    yellow_count = np.sum(yellow_mask)
    red_count = np.sum(red_mask)

    stats_text = (
        f'Total points: {total_points}\n'
        f'Within both: {green_count} ({green_count/total_points*100:.1f}%)\n'
        f'X only: {blue_count} ({blue_count/total_points*100:.1f}%)\n'
        f'Y only: {yellow_count} ({yellow_count/total_points*100:.1f}%)\n'
        f'Outside both: {red_count} ({red_count/total_points*100:.1f}%)'
    )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    return fig