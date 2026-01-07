import polars as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import gaussian_kde
import argparse
import sys
import os
<<<<<<< HEAD
from pathlib import Path
import time
=======
>>>>>>> refs/remotes/origin/main

def tabular_sorted_search(
        alt_data: pl.DataFrame,
        alt_data_col: str,
        ref_data: pl.DataFrame,
        ref_data_col: str,
) -> pl.DataFrame:
    """
    Tabular search for a reference data set in an alternative data set.
    
    Parameters
    ----------
    alt_data : pl.DataFrame
        The alternative data set.
    alt_data_col : str
        The column name in the alternative data set.
    ref_data : pl.DataFrame
        The reference data set.
    ref_data_col : str
        The column name in the reference data set.
    
    Returns
    -------
    pl.DataFrame
        The results of the search.
    """

    # Sort ref_data to ensure binary search works
    ref_data = ref_data.sort(pl.col(ref_data_col), multithreaded=True)

    # Do not sort alt_data because we want to preserve the original order

<<<<<<< HEAD
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


def plot_tolerance_histogram(series: pl.Series, tolerance: float,
                            x_label: str = None, title: str = None,
                            figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Create an academic-style histogram with tolerance-based color coding.
    
    Parameters:
    - series: Data series to plot
    - tolerance: Tolerance value for grouping
    - x_label: Label for x-axis
    - title: Plot title
    - figsize: Figure size
    
    Returns:
    - Figure object
    """
    # Filter out null values
    valid_data = series.drop_nulls()
    
    if len(valid_data) == 0:
        # If no valid data, create empty plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No valid data to plot',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax.set_xlabel(x_label or series.name or 'Value', fontsize=12, fontweight='bold')
        ax.set_title(title or 'Distribution Analysis', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    # Convert to numpy for easier manipulation
    data = valid_data.to_numpy()
    
    # Create bins based on tolerance
    max_val = np.max(np.abs(data))
    if tolerance > 0:
        bin_width = tolerance
        num_bins = min(7, int(np.ceil(max_val / bin_width)) * 2 + 1)  # Ensure odd number of bins centered at 0
        bins = np.linspace(-num_bins//2 * bin_width, (num_bins//2) * bin_width, num_bins+1)
    else:
        # If tolerance is 0 or negative, create 7 bins automatically
        bins = np.linspace(-max_val, max_val, 8)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for each bin
    colors = plt.cm.tab10(np.linspace(0, 1, len(bins)-1))
    
    # Create histogram with custom colors
    n, bins_edges, patches = ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    
    # Color each bar differently
    for i, patch in enumerate(patches):
        patch.set_facecolor(colors[i])
    
    # Add tolerance line if tolerance is positive
    if tolerance > 0:
        ax.axvline(x=tolerance, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Tolerance = ±{tolerance:.2f}')
        ax.axvline(x=-tolerance, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Calculate statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    within_tolerance = np.sum(np.abs(data) <= tolerance) if tolerance > 0 else 0
    within_tolerance_pct = (within_tolerance / len(data)) * 100 if len(data) > 0 else 0
    
    # Add statistics text box
    stats_text = (
        f'Total points: {len(data)}\n'
        f'Mean: {mean_val:.2f}\n'
        f'Std Dev: {std_val:.2f}\n'
        f'Within tolerance: {within_tolerance} ({within_tolerance_pct:.1f}%)'
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Set labels and title
    ax.set_xlabel(x_label or series.name or 'Value', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Distribution Analysis', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add legend positioned to avoid overlapping with the plot
    if tolerance > 0:
        ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    
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
    """Create comparison histogram plots and save them."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate differences
    diff_data = calculate_differences(matched_data)
    
    # Create plots for each relevant field
    plots = [
        ('ppm_diff_by_mz', tolerance_ppm, 'PPM Differences (by MZ)', 'ppm_diff_by_mz.png'),
        ('ppm_diff_by_rt', tolerance_ppm, 'PPM Differences (by RT)', 'ppm_diff_by_rt.png'),
        ('dalton_diff_by_mz', tolerance_mz, 'Dalton Differences (by MZ)', 'dalton_diff_by_mz.png'),
        ('dalton_diff_by_rt', tolerance_mz, 'Dalton Differences (by RT)', 'dalton_diff_by_rt.png'),
        ('rt_diff_by_mz', tolerance_rt, 'RT Differences (by MZ)', 'rt_diff_by_mz.png'),
        ('rt_diff_by_rt', tolerance_rt, 'RT Differences (by RT)', 'rt_diff_by_rt.png'),
    ]
    
    for col, tolerance, title, filename in plots:
        if col in diff_data.columns:
            fig = plot_tolerance_histogram(
                diff_data[col],
                tolerance,
                x_label=col.replace('_', ' ').title(),
                title=title
            )
            fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
=======
    upper_bound = len(ref_data)
    ref_values = ref_data[ref_data_col].to_list()
    
    new_index_list = []

    # Iterate through alt_data column values
    for current_item in alt_data[alt_data_col]:
        
        # search_sorted returns the index where the item should be inserted to maintain order
        sorted_entry_index = ref_data[ref_data_col].search_sorted(current_item)

        if sorted_entry_index == 0: 
            # This is the case where the current alt mz would be the first item in the (re)sorted list
            index_of_closest_match = 0 
        elif sorted_entry_index == upper_bound: 
            # This is the case where the current alt mz would be the last item in the (re)sorted list
            index_of_closest_match = upper_bound - 1 
        else:
            # Check neighbors to find the closest one
            left_val = ref_values[sorted_entry_index - 1]
            right_val = ref_values[sorted_entry_index]
            
            left_diff = abs(left_val - current_item)
            right_diff = abs(right_val - current_item)
            
            if left_diff < right_diff:
                index_of_closest_match = sorted_entry_index - 1
            else:
                index_of_closest_match = sorted_entry_index
>>>>>>> refs/remotes/origin/main

        new_index_list.append(index_of_closest_match)

    # Select the matching rows from ref_data
    matched_ref_data = ref_data[new_index_list, :]
    
    # Combine alt_data and matched_ref_data
    # We want to keep alt_data structure and add ref_data columns horizontally
    # Since we preserved order, we can just hstack if lengths match (which they should)
    
    # Rename ref columns to avoid collision if necessary, or just let polars handle it (it might error on duplicate names)
    # For safety, let's prefix ref columns if they exist in alt
    
    # Actually, let's just use hstack. Polars handles duplicate names by suffixing? No, it errors usually.
    # Let's rename ref columns to have a prefix "ref_"
    
    matched_ref_data = matched_ref_data.select(
        [pl.col(c).alias(f"ref_{c}") for c in matched_ref_data.columns]
    )
    
    result = pl.concat([alt_data, matched_ref_data], how="horizontal")
    
    return result

def calculate_differences(
    data: pl.DataFrame,
    alt_col: str,
    ref_col: str,
    calculate_ppm: bool = False
) -> pl.DataFrame:
    """
    Calculates the absolute difference between the alt and ref columns.
    Optionally calculates PPM difference.
    Assumes the ref column has been renamed to 'ref_{ref_col}' by tabular_sorted_search.
    """
    ref_col_name = f"ref_{ref_col}"
    
    if ref_col_name not in data.columns:
        raise ValueError(f"Expected column {ref_col_name} not found in result dataframe.")
        
    data = data.with_columns(
        (pl.col(alt_col) - pl.col(ref_col_name)).abs().alias("diff")
    )
    
    if calculate_ppm:
        # PPM = (diff / ref_mz) * 10^6
        # We use the ref value as the denominator
        data = data.with_columns(
            ((pl.col("diff") / pl.col(ref_col_name)) * 1e6).alias("diff_in_ppm")
        )
        
    return data

def sort_results(
    data: pl.DataFrame,
    sort_col: str
) -> pl.DataFrame:
    """
    Sorts the dataframe by the specified column in ascending order.
    """
    if sort_col not in data.columns:
        print(f"Warning: Column {sort_col} not found for sorting. Skipping sort.")
        return data
        
    return data.sort(sort_col)

def plot_differences(
    data: pl.DataFrame,
    output_path: str,
    column_to_plot: str = "diff",
    img_fmt: str = "png",
    ppm_marker: float = 15.0
):
    """
    Generates a plot of the specified column.
    If PPM: Density plot with V-Line marker.
    If Absolute: Histogram with max 7 bins, colored bars, and legend.
    """
    if column_to_plot not in data.columns:
        print(f"Warning: Column {column_to_plot} not found for plotting.")
        return

    values = data[column_to_plot].to_numpy()
    
    plt.figure(figsize=(10, 6))
    
    if "ppm" in column_to_plot.lower():
        # Density Plot for PPM
        try:
            density = gaussian_kde(values)
            xs = np.linspace(min(values), max(values), 200)
            plt.plot(xs, density(xs), label='Density')
            plt.fill_between(xs, density(xs), alpha=0.3)
            
            # Add V-Line marker
            plt.axvline(x=ppm_marker, color='red', linestyle='--', label=f'Marker ({ppm_marker} ppm)')
            
            plt.title(f"Density Plot of Differences (PPM)")
            plt.xlabel(f"Difference ({column_to_plot})")
            plt.ylabel("Density")
            plt.legend()
            
        except Exception as e:
            print(f"Error generating density plot: {e}. Falling back to histogram.")
            plt.hist(values, bins=30, edgecolor='black', alpha=0.7)
            plt.title(f"Histogram of Differences (PPM)")
            
    else:
        # Histogram for Absolute Diff (existing logic)
        # Calculate histogram bins and counts first to set up colors
        # We want max 7 bins.
        n, bins, patches = plt.hist(values, bins=7, edgecolor='black')
        
        # Apply colors to bars
        # Use plt.get_cmap which is a valid alternative to cm.get_cmap
        colormap = plt.get_cmap('viridis', len(patches))
        
        for i, patch in enumerate(patches):
            color = colormap(i)
            patch.set_facecolor(color)
            
        # Create legend
        # Each bin range: [bins[i], bins[i+1])
        handles = []
        for i in range(len(patches)):
            color = colormap(i)
            label = f"{bins[i]:.2f} - {bins[i+1]:.2f}"
            handles.append(plt.Rectangle((0,0),1,1, color=color, label=label))
            
        plt.legend(handles=handles, title=f"{column_to_plot} Ranges")
        
        plt.title(f"Histogram of Differences (Absolute)")
        plt.xlabel(f"Difference ({column_to_plot})")
        plt.ylabel("Count")

    plt.grid(axis='y', alpha=0.75)
    
    full_output_path = f"{output_path}_plot.{img_fmt}"
    plt.savefig(full_output_path, format=img_fmt)
    plt.close()
    print(f"Plot saved to {full_output_path}")

def load_data(file_path: str, delimiter: str = None) -> pl.DataFrame:
    """
    Loads data from a file. Auto-detects delimiter if not provided, 
    but prefers explicit delimiter if given.
    """
    if delimiter is None:
        # Simple heuristic based on extension
        if file_path.endswith('.csv'):
            delimiter = ','
        elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
            delimiter = '\t'
        else:
            delimiter = '\t' # Default fallback
            
    try:
        return pl.read_csv(file_path, separator=delimiter)
    except Exception as e:
        print(f"Error reading {file_path} with delimiter '{delimiter}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Mass Spec Comparison Workflow")
    
    parser.add_argument("--ref", required=True, help="Path to reference data file")
    parser.add_argument("--alt", required=True, help="Path to alternative data file")
    parser.add_argument("--ref-col", required=True, help="Column name in reference data to compare")
    parser.add_argument("--alt-col", required=True, help="Column name in alternative data to compare")
    
<<<<<<< HEAD
    # Optional arguments
    parser.add_argument('--mz-tolerance', type=float, default=None,
                       help='Absolute tolerance for m/z matching (Da)')
    parser.add_argument('--ppm-tolerance', type=float, default=5.0,
                       help='Relative tolerance for m/z matching (ppm - parts per million, default: 5 ppm)')
    parser.add_argument('--rt-tolerance', type=float, default=0.5,
                       help='Absolute tolerance for retention time matching (min, default: 0.5 min = 30 sec)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path for results (default: stdout)')
    parser.add_argument('--formats', nargs='+', choices=['csv', 'tsv', 'parquet', 'excel'],
                       default=['csv'], help='Output formats to generate')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress informational output')
    parser.add_argument('--concise', action='store_true', default=True,
                       help='Include only relevant fields in the output (default: True)')
    parser.add_argument('--interactive', action='store_true', default=False,
                       help='Run in interactive mode with Qt interface (default: False)')
    
    args = parser.parse_args()
    
    # Handle interactive mode
    if args.interactive:
        # Placeholder for interactive Qt interface
        if not args.quiet:
            print("Interactive mode is not yet implemented.")
        # In a full implementation, this would launch the Qt interface
        return
    
    try:
        # Create a timestamped subdirectory
        timestamp = int(time.time())
        output_dir = Path("mass_comp_output") / str(timestamp)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a log file with the parameters
        log_file = output_dir / "parameters.log"
        with open(log_file, "w") as f:
            f.write(f"Parameters used:\n")
            f.write(f"Alternative data: {args.alt_data}\n")
            f.write(f"Ground truth data: {args.ground_truth}\n")
            f.write(f"MZ tolerance: {args.mz_tolerance}\n")
            f.write(f"PPM tolerance: {args.ppm_tolerance}\n")
            f.write(f"RT tolerance: {args.rt_tolerance}\n")
            f.write(f"Concise mode: {args.concise}\n")
            f.write(f"Interactive mode: {args.interactive}\n")
        
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
        
        # Apply concise mode if enabled
        if args.concise:
            # Select only relevant columns
            relevant_cols = [
                "mz", "rt", "mz_by_mz", "rt_by_mz", "mz_by_rt", "rt_by_rt",
                "ppm_diff_by_mz", "dalton_diff_by_mz", "rt_diff_by_mz",
                "ppm_diff_by_rt", "dalton_diff_by_rt", "rt_diff_by_rt"
            ]
            diff_data = diff_data.select([col for col in relevant_cols if col in diff_data.columns])
        
        # Save results
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
            plots_dir = output_dir / 'plots'
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
=======
    parser.add_argument("--ref-sep", default=None, help="Delimiter for reference file (default: auto/tab)")
    parser.add_argument("--alt-sep", default=None, help="Delimiter for alternative file (default: auto/comma)")
    
    parser.add_argument("--ppm", action="store_true", help="Calculate and plot differences in PPM")
    parser.add_argument("--ppm-marker", type=float, default=15.0, help="Marker value for PPM density plot (default: 15.0)")
    
    parser.add_argument("--output", required=True, help="Base name for output files")
    parser.add_argument("--img-fmt", default="png", help="Image format (default: png)")
    
    args = parser.parse_args()
    
    print("Loading data...")
    ref_data = load_data(args.ref, args.ref_sep)
    alt_data = load_data(args.alt, args.alt_sep)
>>>>>>> refs/remotes/origin/main
    
    print("Performing comparison...")
    # Verify columns exist
    if args.ref_col not in ref_data.columns:
        print(f"Error: Column '{args.ref_col}' not found in reference data.")
        sys.exit(1)
    if args.alt_col not in alt_data.columns:
        print(f"Error: Column '{args.alt_col}' not found in alternative data.")
        sys.exit(1)
        
    matched_data = tabular_sorted_search(alt_data, args.alt_col, ref_data, args.ref_col)
    
    print("Calculating differences...")
    final_data = calculate_differences(matched_data, args.alt_col, args.ref_col, calculate_ppm=args.ppm)
    
    # Sort results
    sort_col = "diff_in_ppm" if args.ppm else "diff"
    print(f"Sorting results by {sort_col}...")
    final_data = sort_results(final_data, sort_col)
    
    # Save table
    output_table_path = f"{args.output}_table.tsv"
    final_data.write_csv(output_table_path, separator='\t')
    print(f"Results table saved to {output_table_path}")
    
    print("Generating plot...")
    plot_col = "diff_in_ppm" if args.ppm else "diff"
    plot_differences(final_data, args.output, column_to_plot=plot_col, img_fmt=args.img_fmt, ppm_marker=args.ppm_marker)
    
    print("Done.")

<<<<<<< HEAD

if __name__ == '__main__':
=======
if __name__ == "__main__":
>>>>>>> refs/remotes/origin/main
    main()
