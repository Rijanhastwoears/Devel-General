import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from pathlib import Path

def match_ground_truth(
    alt_data: pl.DataFrame,
    ground_truth: pl.DataFrame,
    *,
    tolerance_mz: float | None = None,
    tolerance_ppm: float | None = None,
    tolerance_rt: float | None = None,
) -> pl.DataFrame:
    """
    Match alternative mass spec rows to ground-truth rows using Polars asof joins.

    For each row in alt_data this finds:
      - nearest ground-truth row by mz (adds mz_by_mz, rt_by_mz)
      - nearest ground-truth row by rt (adds mz_by_rt, rt_by_rt)

    Optional tolerances (absolute) can be passed; if the nearest neighbour is
    further than the tolerance the matched columns will be null.

    Parameters:
    - tolerance_mz: Absolute m/z tolerance in Daltons (Da)
    - tolerance_ppm: Relative m/z tolerance in parts per million (ppm)
    - tolerance_rt: Absolute retention time tolerance in minutes

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

    # Validate that only one tolerance type is specified
    if tolerance_mz is not None and tolerance_ppm is not None:
        raise ValueError("Cannot specify both tolerance_mz and tolerance_ppm")

    # Prepare ground-truth frames sorted for asof joins; alias columns to produce clear result names
    gt_by_mz = gt.sort("mz").select([pl.col("mz").alias("mz_by_mz"), pl.col("rt").alias("rt_by_mz"), pl.col("mz")])
    gt_by_rt = gt.sort("rt").select([pl.col("mz").alias("mz_by_rt"), pl.col("rt").alias("rt_by_rt"), pl.col("rt")])

    # Add index to restore original order later
    alt_indexed = alt.with_row_index("_alt_idx")

    # As-of join by mz (sort by mz on the left for join_asof)
    if tolerance_ppm is not None:
        # For ppm tolerance, we need to join without tolerance first, then filter by ppm
        # since asof join doesn't support row-specific tolerances
        joined_mz_idx = (
            alt_indexed.sort("mz")
            .join_asof(
                gt_by_mz,
                left_on="mz",
                right_on="mz",
                strategy="nearest",
            )
            .filter((abs(pl.col("mz") - pl.col("mz_by_mz")) / pl.col("mz_by_mz") * 1_000_000) <= tolerance_ppm)
            .select("_alt_idx", "mz_by_mz", "rt_by_mz")
        )
    else:
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
    Shows the full data range with tolerance highlighting.

    Color scheme:
      - Green: Below both x_tol and y_tol (good matches)
      - Blue: Below x_tol only
      - Yellow: Below y_tol only
      - Red: Above both tolerances (poor matches)
    """
    # Filter out null values that can cause NaN axis limits
    valid_mask = x_series.is_not_null() & y_series.is_not_null()
    valid_x = x_series.filter(valid_mask)
    valid_y = y_series.filter(valid_mask)
    
    if len(valid_x) == 0 or len(valid_y) == 0:
        # If no valid data, create empty plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No valid data to plot',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax.set_xlabel(x_label or x_series.name or 'X', fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label or y_series.name or 'Y', fontsize=12, fontweight='bold')
        ax.set_title(title or 'Tolerance Analysis', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    x_data = valid_x.to_numpy()
    y_data = valid_y.to_numpy()

    below_x = np.abs(x_data) < x_tol
    below_y = np.abs(y_data) < y_tol

    green_mask = below_x & below_y
    blue_mask = below_x & ~below_y
    yellow_mask = ~below_x & below_y
    red_mask = ~below_x & ~below_y

    fig, ax = plt.subplots(figsize=figsize)

    # Plot points in order to ensure proper layering
    if np.any(red_mask):
        ax.scatter(x_data[red_mask], y_data[red_mask],
                   c='red', alpha=0.6, s=50, label='Outside both tolerances',
                   edgecolors='darkred', linewidth=0.5, zorder=1)
    
    if np.any(blue_mask):
        ax.scatter(x_data[blue_mask], y_data[blue_mask],
                   c='blue', alpha=0.6, s=50, label='Within x-tolerance only',
                   edgecolors='darkblue', linewidth=0.5, zorder=2)

    if np.any(yellow_mask):
        ax.scatter(x_data[yellow_mask], y_data[yellow_mask],
                   c='gold', alpha=0.6, s=50, label='Within y-tolerance only',
                   edgecolors='orange', linewidth=0.5, zorder=2)
    
    if np.any(green_mask):
        ax.scatter(x_data[green_mask], y_data[green_mask],
                   c='green', alpha=0.6, s=50, label='Within both tolerances',
                   edgecolors='darkgreen', linewidth=0.5, zorder=3)

    # Calculate data ranges for setting appropriate limits
    x_range = x_data.max() - x_data.min()
    y_range = y_data.max() - y_data.min()
    x_margin = x_range * 0.05  # 5% margin
    y_margin = y_range * 0.05  # 5% margin

    # Set axis limits to show full data range
    ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
    ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)

    # Add tolerance lines and shaded regions
    if not np.isinf(x_tol):
        ax.axvline(x=x_tol, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'x-tolerance = ±{x_tol:.2f}')
        ax.axvline(x=-x_tol, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvspan(-x_tol, x_tol, alpha=0.1, color='green', zorder=0)

    if not np.isinf(y_tol):
        ax.axhline(y=y_tol, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'y-tolerance = ±{y_tol:.2f}')
        ax.axhline(y=-y_tol, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhspan(-y_tol, y_tol, alpha=0.1, color='green', zorder=0)

    ax.set_xlabel(x_label or x_series.name or 'X', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label or y_series.name or 'Y', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Tolerance Analysis', fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)

    # Position stats box in upper left corner to avoid overlapping with tolerance lines
    total_points = len(valid_x)  # Use the valid (non-null) data count
    green_count = np.sum(green_mask)
    blue_count = np.sum(blue_mask)
    yellow_count = np.sum(yellow_mask)
    red_count = np.sum(red_mask)
    
    null_count = len(x_series) - total_points  # Count of null values filtered out

    stats_text = (
        f'Total points: {len(x_series)}\n'
        f'Valid points: {total_points}\n'
        f'Null points: {null_count}\n'
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


def load_dataframe(file_path: str) -> pl.DataFrame:
    """Load a DataFrame from various file formats."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.csv':
        return pl.read_csv(file_path)
    elif suffix == '.tsv':
        return pl.read_csv(file_path, separator='\t')
    elif suffix in ['.parquet', '.pq']:
        return pl.read_parquet(file_path)
    elif suffix in ['.xlsx', '.xls']:
        return pl.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .csv, .tsv, .parquet, .xlsx")


def save_dataframe(df: pl.DataFrame, output_path: str) -> None:
    """Save a DataFrame to various file formats."""
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = output_path.suffix.lower()
    
    if suffix == '.csv':
        df.write_csv(output_path)
    elif suffix == '.tsv':
        df.write_csv(output_path, separator='\t')
    elif suffix in ['.parquet', '.pq']:
        df.write_parquet(output_path)
    elif suffix in ['.xlsx', '.xls']:
        df.write_excel(output_path)
    else:
        raise ValueError(f"Unsupported output format: {suffix}. Supported formats: .csv, .tsv, .parquet, .xlsx")


def create_comparison_plots(matched_data: pl.DataFrame,
                          tolerance_mz: float,
                          tolerance_ppm: float,
                          tolerance_rt: float,
                          output_dir: str) -> None:
    """Create comparison plots and save them."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate differences
    diff_data = calculate_differences(matched_data)
    
    # Create plots
    plots = [
        ('ppm_diff_by_mz', 'rt_diff_by_mz', 'PPM vs RT Differences (by MZ)', 'mz_tolerance_plots.png'),
        ('ppm_diff_by_rt', 'rt_diff_by_rt', 'PPM vs RT Differences (by RT)', 'rt_tolerance_plots.png'),
        ('dalton_diff_by_mz', 'rt_diff_by_mz', 'Dalton vs RT Differences (by MZ)', 'dalton_rt_mz_plots.png'),
        ('dalton_diff_by_rt', 'rt_diff_by_rt', 'Dalton vs RT Differences (by RT)', 'dalton_rt_rt_plots.png'),
    ]
    
    for x_col, y_col, title, filename in plots:
        if x_col in diff_data.columns and y_col in diff_data.columns:
            # Determine appropriate tolerance for x-axis
            if 'ppm' in x_col:
                x_tol = tolerance_ppm if tolerance_ppm is not None else float('inf')
            elif 'dalton' in x_col:
                x_tol = tolerance_mz
            else:
                x_tol = tolerance_rt
            
            fig = plot_tolerance_scatter(
                diff_data[x_col],
                diff_data[y_col],
                x_tol,
                tolerance_rt,
                x_label=x_col.replace('_', ' ').title(),
                y_label=y_col.replace('_', ' ').title(),
                title=title
            )
            fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            plt.close(fig)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Compare mass spectrometry feature lists with ground truth data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s alt_data.csv ground_truth.csv
  %(prog)s alt_data.tsv ground_truth.tsv
  %(prog)s alt_data.csv ground_truth.csv --mz-tolerance 0.01 --rt-tolerance 0.5
  %(prog)s alt_data.csv ground_truth.csv --output results.tsv --formats tsv
  %(prog)s alt_data.parquet ground_truth.parquet --output-dir results/ --formats csv parquet
        '''
    )
    
    # Positional arguments
    parser.add_argument('alt_data', help='Path to alternative data file (CSV, TSV, Parquet, Excel)')
    parser.add_argument('ground_truth', help='Path to ground truth data file (CSV, TSV, Parquet, Excel)')
    
    # Optional arguments
    parser.add_argument('--mz-tolerance', type=float, default=None,
                       help='Absolute tolerance for m/z matching (Da)')
    parser.add_argument('--ppm-tolerance', type=float, default=5.0,
                       help='Relative tolerance for m/z matching (ppm - parts per million, default: 5 ppm)')
    parser.add_argument('--rt-tolerance', type=float, default=0.5,
                       help='Absolute tolerance for retention time matching (min, default: 0.5 min = 30 sec)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path for results (default: stdout)')
    parser.add_argument('--output-dir', '-d', type=str, default='.',
                       help='Output directory for results and plots (default: current directory)')
    parser.add_argument('--formats', nargs='+', choices=['csv', 'tsv', 'parquet', 'excel'],
                       default=['csv'], help='Output formats to generate')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--plots-dir', type=str, default='plots',
                       help='Directory for plots (default: plots)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress informational output')
    
    args = parser.parse_args()
    
    try:
        # Load data
        if not args.quiet:
            print(f"Loading alternative data from {args.alt_data}...")
        alt_data = load_dataframe(args.alt_data)
        
        if not args.quiet:
            print(f"Loading ground truth data from {args.ground_truth}...")
        ground_truth = load_dataframe(args.ground_truth)
        
        # Validate required columns
        required_cols = ['mz', 'rt']
        missing_alt = [col for col in required_cols if col not in alt_data.columns]
        missing_gt = [col for col in required_cols if col not in ground_truth.columns]
        
        if missing_alt:
            raise ValueError(f"Alternative data missing required columns: {missing_alt}")
        if missing_gt:
            raise ValueError(f"Ground truth data missing required columns: {missing_gt}")
        
        if not args.quiet:
            print(f"Alternative data shape: {alt_data.shape}")
            print(f"Ground truth data shape: {ground_truth.shape}")
        
        # Perform matching
        if not args.quiet:
            print("Performing nearest-neighbor matching...")
        matched_data = match_ground_truth(
            alt_data,
            ground_truth,
            tolerance_mz=args.mz_tolerance,
            tolerance_ppm=args.ppm_tolerance,
            tolerance_rt=args.rt_tolerance
        )
        
        # Calculate differences
        if not args.quiet:
            print("Calculating differences...")
        diff_data = calculate_differences(matched_data)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for fmt in args.formats:
            if fmt == 'csv':
                output_file = output_dir / 'results.csv'
                diff_data.write_csv(output_file)
            elif fmt == 'tsv':
                output_file = output_dir / 'results.tsv'
                diff_data.write_csv(output_file, separator='\t')
            elif fmt == 'parquet':
                output_file = output_dir / 'results.parquet'
                diff_data.write_parquet(output_file)
            elif fmt == 'excel':
                output_file = output_dir / 'results.xlsx'
                diff_data.write_excel(output_file)
            
            if not args.quiet:
                print(f"Results saved to {output_file}")
        
        # Generate plots
        if not args.no_plots:
            if not args.quiet:
                print("Generating comparison plots...")
            plots_dir = output_dir / args.plots_dir
            create_comparison_plots(diff_data, args.mz_tolerance or 0, args.ppm_tolerance or 0, args.rt_tolerance or 0, str(plots_dir))
            
            if not args.quiet:
                print(f"Plots saved to {plots_dir}")
        
        # Print summary statistics
        if not args.quiet:
            print("\nSummary Statistics:")
            print(f"Total alternative data rows: {len(diff_data)}")
            
            # Count matches that actually have valid matches (non-null mz_by_mz and rt_by_mz)
            valid_mz_matches = diff_data.filter(pl.col("mz_by_mz").is_not_null())
            valid_rt_matches = diff_data.filter(pl.col("rt_by_mz").is_not_null())
            valid_both_matches = diff_data.filter(pl.col("mz_by_mz").is_not_null() & pl.col("rt_by_mz").is_not_null())
            
            print(f"Rows with valid m/z matches: {len(valid_mz_matches)} ({len(valid_mz_matches)/len(diff_data)*100:.1f}%)")
            print(f"Rows with valid RT matches: {len(valid_rt_matches)} ({len(valid_rt_matches)/len(diff_data)*100:.1f}%)")
            print(f"Rows with both m/z and RT matches: {len(valid_both_matches)} ({len(valid_both_matches)/len(diff_data)*100:.1f}%)")
            
            # Count matches within specified tolerances
            if args.mz_tolerance is not None:
                mz_tol = args.mz_tolerance
                within_mz = valid_mz_matches.filter(pl.col("dalton_diff_by_mz") <= mz_tol)
                print(f"Matches within m/z tolerance ({mz_tol} Da): {len(within_mz)} ({len(within_mz)/len(diff_data)*100:.1f}%)")
            elif args.ppm_tolerance is not None:
                ppm_tol = args.ppm_tolerance
                within_ppm = valid_mz_matches.filter(pl.col("ppm_diff_by_mz") <= ppm_tol)
                print(f"Matches within PPM tolerance ({ppm_tol} ppm): {len(within_ppm)} ({len(within_ppm)/len(diff_data)*100:.1f}%)")
            else:
                print("No m/z or PPM tolerance specified")
            
            if args.rt_tolerance is not None:
                rt_tol = args.rt_tolerance
                within_rt = valid_rt_matches.filter(pl.col("rt_diff_by_mz") <= rt_tol)
                print(f"Matches within RT tolerance ({rt_tol} min): {len(within_rt)} ({len(within_rt)/len(diff_data)*100:.1f}%)")
            else:
                print("No RT tolerance specified")
            
            # Summary of matches meeting all specified tolerances
            fully_within_tolerance = valid_both_matches
            
            if args.mz_tolerance is not None:
                fully_within_tolerance = fully_within_tolerance.filter(pl.col("dalton_diff_by_mz") <= args.mz_tolerance)
            elif args.ppm_tolerance is not None:
                fully_within_tolerance = fully_within_tolerance.filter(pl.col("ppm_diff_by_mz") <= args.ppm_tolerance)
                
            if args.rt_tolerance is not None:
                fully_within_tolerance = fully_within_tolerance.filter(pl.col("rt_diff_by_mz") <= args.rt_tolerance)
            
            if args.mz_tolerance is not None or args.ppm_tolerance is not None or args.rt_tolerance is not None:
                print(f"Matches meeting all specified tolerances: {len(fully_within_tolerance)} ({len(fully_within_tolerance)/len(diff_data)*100:.1f}%)")
            else:
                print("No tolerances specified for comprehensive comparison")
            
            if 'ppm_diff_by_mz' in diff_data.columns and len(valid_mz_matches) > 0:
                print(f"Mean PPM difference (by mz): {valid_mz_matches['ppm_diff_by_mz'].mean():.2f}")
            if 'rt_diff_by_mz' in diff_data.columns and len(valid_rt_matches) > 0:
                print(f"Mean RT difference (by mz): {valid_rt_matches['rt_diff_by_mz'].mean():.2f}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()