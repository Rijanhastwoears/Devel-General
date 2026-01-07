import polars as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import gaussian_kde
import argparse
import sys
import os

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

if __name__ == "__main__":
    main()
