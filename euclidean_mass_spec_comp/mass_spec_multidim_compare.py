import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import argparse
import sys
from typing import List, Tuple, Dict, Optional

def load_data(file_path: str, delimiter: str = None) -> pl.DataFrame:
    """
    Loads data from a file. Auto-detects delimiter if not provided.
    """
    if delimiter is None:
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

def normalize_data(
    data: pl.DataFrame, 
    columns: List[str], 
    stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[pl.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Normalizes the specified columns using Z-score normalization.
    If stats are provided, uses them. Otherwise, calculates and returns them.
    
    Returns:
        Tuple containing:
        - DataFrame with normalized columns (prefixed with 'norm_')
        - Dictionary of statistics {col_name: (mean, std)}
    """
    normalized_data = data.clone()
    computed_stats = {}
    
    for col in columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")
            
        values = data[col].to_numpy()
        
        if stats and col in stats:
            mean, std = stats[col]
        else:
            mean = np.mean(values)
            std = np.std(values)
            # Avoid division by zero
            if std == 0:
                std = 1.0
            computed_stats[col] = (mean, std)
            
        # Create normalized column
        norm_col_name = f"norm_{col}"
        normalized_values = (values - mean) / std
        
        normalized_data = normalized_data.with_columns(
            pl.Series(norm_col_name, normalized_values)
        )
        
    return normalized_data, stats if stats else computed_stats

def build_kdtree(data: pl.DataFrame, columns: List[str]) -> KDTree:
    """
    Builds a K-d tree from the specified normalized columns.
    """
    # Extract data for the tree
    # Assumes columns are already normalized and exist
    tree_data = data.select(columns).to_numpy()
    return KDTree(tree_data)

def query_kdtree(
    tree: KDTree, 
    query_data: pl.DataFrame, 
    query_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Queries the K-d tree for nearest neighbors.
    Returns distances and indices of the nearest neighbors in the tree's data.
    """
    query_points = query_data.select(query_columns).to_numpy()
    distances, indices = tree.query(query_points, k=1)
    return distances, indices

def assemble_results(
    alt_data: pl.DataFrame, 
    ref_data: pl.DataFrame, 
    indices: np.ndarray, 
    distances: np.ndarray,
    dim_mappings: List[Tuple[str, str, str]]
) -> pl.DataFrame:
    """
    Assembles the final result DataFrame.
    """
    # Get the matched rows from ref_data using indices
    matched_ref = ref_data[indices, :]
    
    # Rename ref columns to avoid collision and indicate source
    matched_ref = matched_ref.select(
        [pl.col(c).alias(f"ref_{c}") for c in matched_ref.columns]
    )
    
    # Combine alt_data and matched_ref
    result = pl.concat([alt_data, matched_ref], how="horizontal")
    
    # Add the calculated distance (in normalized space)
    result = result.with_columns(pl.Series("normalized_distance", distances))
    
    # Calculate absolute differences for each dimension
    for name, ref_col, alt_col in dim_mappings:
        ref_col_name = f"ref_{ref_col}"
        diff_col_name = f"diff_{name}"
        
        result = result.with_columns(
            (pl.col(alt_col) - pl.col(ref_col_name)).abs().alias(diff_col_name)
        )
        
    return result

def plot_correspondence(
    data: pl.DataFrame, 
    dims: List[Tuple[str, str, str]], 
    output_path: str
):
    """
    Generates a correspondence scatter plot for the first two dimensions.
    Ref points: Green Circles ('go')
    Alt points: Blue Triangles ('b^')
    Lines connect matched pairs.
    """
    if len(dims) < 2:
        print("Warning: Need at least 2 dimensions for scatter plot. Skipping plot.")
        return

    # Use first two dimensions for plotting
    x_name, x_ref_col, x_alt_col = dims[0]
    y_name, y_ref_col, y_alt_col = dims[1]
    
    # Ref columns in result are prefixed with 'ref_'
    x_ref_res = f"ref_{x_ref_col}"
    y_ref_res = f"ref_{y_ref_col}"
    
    # Extract data
    # We need to be careful with column names if they are the same in alt and ref (which they often are)
    # The result df has alt columns as original names, and ref columns as 'ref_' + original
    
    alt_x = data[x_alt_col].to_numpy()
    alt_y = data[y_alt_col].to_numpy()
    
    ref_x = data[x_ref_res].to_numpy()
    ref_y = data[y_ref_res].to_numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Plot connections first so points are on top
    # We can plot all lines at once using a collection or just a loop
    # For matplotlib, plotting many lines can be slow if done individually. 
    # But for typical mass spec comparison (thousands of points), it might be okay.
    # Let's use plot with None separators for efficiency if needed, or just a loop for simplicity first.
    # Actually, plotting segments is better.
    
    # Create segments: [(x1, y1), (x2, y2)]
    # We can plot all lines as a single plot command with None in between to break lines
    # x_coords = [x1, x2, None, x3, x4, None, ...]
    
    x_coords = []
    y_coords = []
    for i in range(len(data)):
        x_coords.extend([alt_x[i], ref_x[i], None])
        y_coords.extend([alt_y[i], ref_y[i], None])
        
    plt.plot(x_coords, y_coords, 'k-', alpha=0.2, linewidth=0.5, label='Match')
    
    # Plot points
    plt.plot(ref_x, ref_y, 'go', alpha=0.6, label='Reference', markersize=6)
    plt.plot(alt_x, alt_y, 'b^', alpha=0.6, label='Alternative', markersize=6)
    
    plt.xlabel(f"{x_name} ({x_alt_col} / {x_ref_col})")
    plt.ylabel(f"{y_name} ({y_alt_col} / {y_ref_col})")
    plt.title(f"Correspondence Plot: {x_name} vs {y_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    full_output_path = f"{output_path}_plot.png"
    plt.savefig(full_output_path, dpi=300)
    plt.close()
    print(f"Correspondence plot saved to {full_output_path}")

def parse_dimensions(dim_args: List[str]) -> List[Tuple[str, str, str]]:
    """
    Parses the --dim arguments.
    Format: name,ref_col,alt_col
    """
    dims = []
    for arg in dim_args:
        parts = arg.split(',')
        if len(parts) != 3:
            print(f"Error: Invalid dimension format '{arg}'. Expected 'name,ref_col,alt_col'.")
            sys.exit(1)
        dims.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return dims

def run_multidim_comparison(
    ref_data: pl.DataFrame,
    alt_data: pl.DataFrame,
    dims: List[Tuple[str, str, str]],
    output_path: str
) -> pl.DataFrame:
    """
    Runs the multi-dimensional comparison workflow.
    
    Parameters
    ----------
    ref_data : pl.DataFrame
        Reference dataset.
    alt_data : pl.DataFrame
        Alternative dataset.
    dims : List[Tuple[str, str, str]]
        List of dimensions to compare. Each tuple is (name, ref_col, alt_col).
    output_path : str
        Base path for output files (table and plot).
        
    Returns
    -------
    pl.DataFrame
        The resulting dataframe with matches and differences.
    """
    ref_cols = [d[1] for d in dims]
    alt_cols = [d[2] for d in dims]
    
    print(f"Comparing {len(dims)} dimensions: {[d[0] for d in dims]}")
    
    # Normalize Ref Data
    print("Normalizing reference data...")
    ref_data_norm, ref_stats = normalize_data(ref_data, ref_cols)
    
    # Normalize Alt Data using Ref Stats
    print("Normalizing alternative data...")
    alt_stats = {}
    for i in range(len(dims)):
        ref_c = ref_cols[i]
        alt_c = alt_cols[i]
        alt_stats[alt_c] = ref_stats[ref_c]
        
    alt_data_norm, _ = normalize_data(alt_data, alt_cols, stats=alt_stats)
    
    # Build K-d Tree on Normalized Ref Data
    print("Building K-d tree...")
    norm_ref_cols = [f"norm_{c}" for c in ref_cols]
    tree = build_kdtree(ref_data_norm, norm_ref_cols)
    
    # Query K-d Tree with Normalized Alt Data
    print("Querying K-d tree...")
    norm_alt_cols = [f"norm_{c}" for c in alt_cols]
    distances, indices = query_kdtree(tree, alt_data_norm, norm_alt_cols)
    
    # Assemble Results
    print("Assembling results...")
    final_data = assemble_results(alt_data, ref_data, indices, distances, dims)
    
    # Save Output
    output_table_path = f"{output_path}.tsv"
    final_data.write_csv(output_table_path, separator='\t')
    print(f"Results saved to {output_table_path}")
    
    # Generate Plot
    print("Generating correspondence plot...")
    plot_correspondence(final_data, dims, output_path)
    
    # Calculate and Save Summary Statistics
    print("Calculating summary statistics...")
    distances = final_data["normalized_distance"]
    total_dist = distances.sum()
    mean_dist = distances.mean()
    median_dist = distances.median()
    
    stats_df = pl.DataFrame({
        "Metric": ["Total Euclidean Distance", "Mean Euclidean Distance", "Median Euclidean Distance"],
        "Value": [total_dist, mean_dist, median_dist]
    })
    
    output_stats_path = f"{output_path}_stats.tsv"
    stats_df.write_csv(output_stats_path, separator='\t')
    print(f"Summary statistics saved to {output_stats_path}")
    
    return final_data

def main():
    parser = argparse.ArgumentParser(description="Multi-dimensional Mass Spec Comparison using K-d Trees")
    
    parser.add_argument("--ref", required=True, help="Path to reference data file")
    parser.add_argument("--alt", required=True, help="Path to alternative data file")
    
    parser.add_argument("--dim", action='append', required=True, 
                        help="Dimension definition: name,ref_col,alt_col (can be used multiple times)")
    
    parser.add_argument("--output", required=True, help="Base name for output file")
    
    parser.add_argument("--ref-sep", default=None, help="Delimiter for reference file")
    parser.add_argument("--alt-sep", default=None, help="Delimiter for alternative file")
    
    args = parser.parse_args()
    
    # Parse dimensions
    dims = parse_dimensions(args.dim)
    
    # Load data
    print("Loading data...")
    ref_data = load_data(args.ref, args.ref_sep)
    alt_data = load_data(args.alt, args.alt_sep)
    
    # Run comparison
    run_multidim_comparison(ref_data, alt_data, dims, args.output)
    
    print("Done.")

if __name__ == "__main__":
    main()
