
def match_ground_truth(alt_data: pl.DataFrame, ground_truth: pl.DataFrame) -> pl.DataFrame:
    """
    Matches alternative mass spectrometry data against ground truth data.
    
    For each row in `alt_data`, this function finds the nearest neighbor in `ground_truth`
    based on two separate criteria:
    1. **m/z value**: It finds the ground truth entry with the closest m/z. The m/z and RT
       of this matched ground truth entry are stored in new columns `mz_by_mz` and `rt_by_mz`.
    2. **RT value**: It finds the ground truth entry with the closest retention time (RT).
       The m/z and RT of this matched ground truth entry are stored in new columns `mz_by_rt`
       and `rt_by_rt`.
    
    Parameters:
    -----------
    alt_data : pl.DataFrame
        A Polars DataFrame containing the alternative data. Must have 'mz' and 'rt' columns.
    ground_truth : pl.DataFrame
        A Polars DataFrame containing the ground truth data. Must have 'mz' and 'rt' columns.
    
    Returns:
    --------
    pl.DataFrame
        The original `alt_data` DataFrame with four new columns appended:
        `mz_by_mz`, `rt_by_mz`, `mz_by_rt`, and `rt_by_rt`, containing the matched ground truth values.
    """
    # Sort ground truth data for binary search
    gt_mz_sorted = ground_truth.sort("mz")
    gt_rt_sorted = ground_truth.sort("rt")
    
    # Convert to lists for iteration
    gt_mz_list = gt_mz_sorted.to_dicts()
    gt_rt_list = gt_rt_sorted.to_dicts()
    
    alt_mz_values = alt_data['mz'].to_list()
    alt_rt_values = alt_data['rt'].to_list()
    
    # Initialize result lists
    mz_by_mz_results = []
    rt_by_mz_results = []
    mz_by_rt_results = []
    rt_by_rt_results = []
    
    # Process each row in alt_data
    for i in range(len(alt_data)):
        current_mz = alt_mz_values[i]
        current_rt = alt_rt_values[i]
        
        # === Match by m/z ===
        mz_search_index = gt_mz_sorted['mz'].search_sorted(current_mz)
        left_index = mz_search_index - 1
        right_index = mz_search_index
        
        # Handle edge cases and find nearest
        if left_index < 0:
            # Only right neighbor exists
            best_mz_match = gt_mz_list[right_index] if right_index < len(gt_mz_list) else None
        elif right_index >= len(gt_mz_list):
            # Only left neighbor exists
            best_mz_match = gt_mz_list[left_index]
        else:
            # Both neighbors exist, find closest
            left_distance = abs(current_mz - gt_mz_list[left_index]['mz'])
            right_distance = abs(current_mz - gt_mz_list[right_index]['mz'])
            best_mz_match = gt_mz_list[left_index] if left_distance < right_distance else gt_mz_list[right_index]
        
        mz_by_mz_results.append(best_mz_match['mz'] if best_mz_match else None)
        rt_by_mz_results.append(best_mz_match['rt'] if best_mz_match else None)
        
        # === Match by RT ===
        rt_search_index = gt_rt_sorted['rt'].search_sorted(current_rt)
        left_index = rt_search_index - 1
        right_index = rt_search_index
        
        # Handle edge cases and find nearest
        if left_index < 0:
            # Only right neighbor exists
            best_rt_match = gt_rt_list[right_index] if right_index < len(gt_rt_list) else None
        elif right_index >= len(gt_rt_list):
            # Only left neighbor exists
            best_rt_match = gt_rt_list[left_index]
        else:
            # Both neighbors exist, find closest
            left_distance = abs(current_rt - gt_rt_list[left_index]['rt'])
            right_distance = abs(current_rt - gt_rt_list[right_index]['rt'])
            best_rt_match = gt_rt_list[left_index] if left_distance < right_distance else gt_rt_list[right_index]
        
        mz_by_rt_results.append(best_rt_match['mz'] if best_rt_match else None)
        rt_by_rt_results.append(best_rt_match['rt'] if best_rt_match else None)
    
    # Add result columns to original dataframe
    result = alt_data.with_columns(
        pl.Series("mz_by_mz", mz_by_mz_results),
        pl.Series("rt_by_mz", rt_by_mz_results),
        pl.Series("mz_by_rt", mz_by_rt_results),
        pl.Series("rt_by_rt", rt_by_rt_results)
    )
    
    return result



def calculate_differences(matched_data: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates the parts per million (ppm) and retention time (RT) differences
    between the alternative mass spectrometry data and its matched ground truth data.

    This function computes four new columns based on the matching criteria:
    - `ppm_diff_by_mz`: The ppm difference between the 'mz' of the alternative data
      and the 'mz_by_mz' (ground truth mz matched by mz).
    - `rt_diff_by_mz`: The absolute RT difference between the 'rt' of the alternative data
      and the 'rt_by_mz' (ground truth rt matched by mz).
    - `ppm_diff_by_rt`: The ppm difference between the 'mz' of the alternative data
      and the 'mz_by_rt' (ground truth mz matched by rt).
    - `rt_diff_by_rt`: The absolute RT difference between the 'rt' of the alternative data
      and the 'rt_by_rt' (ground truth rt matched by rt).

    The ppm difference is calculated using the formula:
    `((abs(measured - theoretical) / theoretical) * 1,000,000)`.

    Parameters:
    -----------
    matched_data : pl.DataFrame
        A Polars DataFrame that is the output of the `match_ground_truth` function.
        It must contain the columns: 'mz', 'rt', 'mz_by_mz', 'rt_by_mz', 'mz_by_rt', 'rt_by_rt'.
    
    Returns:
    --------
    pl.DataFrame
        Input dataframe with additional difference columns
    """
    result = matched_data.with_columns(
        # PPM difference when matched by m/z
        ((abs(pl.col("mz") - pl.col("mz_by_mz")) / pl.col("mz_by_mz")) * 1_000_000).alias("ppm_diff_by_mz"),
        
        # RT difference when matched by m/z
        (pl.col("rt") - pl.col("rt_by_mz")).abs().alias("rt_diff_by_mz"),
        
        # PPM difference when matched by RT
        ((abs(pl.col("mz") - pl.col("mz_by_rt")) / pl.col("mz_by_rt")) * 1_000_000).alias("ppm_diff_by_rt"),
        
        # RT difference when matched by RT
        (pl.col("rt") - pl.col("rt_by_rt")).abs().alias("rt_diff_by_rt")
    )
    
    return result

import matplotlib.pyplot as plt
import numpy as np


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
    
    Parameters:
    -----------
    x_series : pl.Series
        Data for x-axis
    y_series : pl.Series
        Data for y-axis
    x_tol : float
        Tolerance threshold for x-axis values
    y_tol : float
        Tolerance threshold for y-axis values
    x_label : str, optional
        Label for x-axis (defaults to series name)
    y_label : str, optional
        Label for y-axis (defaults to series name)
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    plt.Figure
        The matplotlib figure object
    """
    # Convert to numpy arrays for easier manipulation
    x_data = x_series.to_numpy()
    y_data = y_series.to_numpy()
    
    # Create boolean masks for tolerance conditions
    below_x = np.abs(x_data) < x_tol
    below_y = np.abs(y_data) < y_tol
    
    # Categorize points
    green_mask = below_x & below_y  # Both tolerances met
    blue_mask = below_x & ~below_y  # Only x tolerance met
    yellow_mask = ~below_x & below_y  # Only y tolerance met
    red_mask = ~below_x & ~below_y  # Neither tolerance met
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each category
    if np.any(green_mask):
        ax.scatter(x_data[green_mask], y_data[green_mask], 
                  c='green', alpha=0.6, s=50, label='Within both tolerances', 
                  edgecolors='darkgreen', linewidth=0.5)
    
    if np.any(blue_mask):
        ax.scatter(x_data[blue_mask], y_data[blue_mask], 
                  c='blue', alpha=0.6, s=50, label=f'Within x-tolerance only', 
                  edgecolors='darkblue', linewidth=0.5)
    
    if np.any(yellow_mask):
        ax.scatter(x_data[yellow_mask], y_data[yellow_mask], 
                  c='gold', alpha=0.6, s=50, label=f'Within y-tolerance only', 
                  edgecolors='orange', linewidth=0.5)
    
    if np.any(red_mask):
        ax.scatter(x_data[red_mask], y_data[red_mask], 
                  c='red', alpha=0.6, s=50, label='Outside both tolerances', 
                  edgecolors='darkred', linewidth=0.5)
    
    # Add tolerance threshold lines
    ax.axvline(x=x_tol, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'x-tolerance = ±{x_tol}')
    ax.axvline(x=-x_tol, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=y_tol, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'y-tolerance = ±{y_tol}')
    ax.axhline(y=-y_tol, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add tolerance box (shaded region where both tolerances are met)
    ax.axvspan(-x_tol, x_tol, alpha=0.1, color='green')
    ax.axhspan(-y_tol, y_tol, alpha=0.1, color='green')
    
    # Labels and title
    ax.set_xlabel(x_label or x_series.name or 'X', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label or y_series.name or 'Y', fontsize=12, fontweight='bold')
    ax.set_title(title or 'Tolerance Analysis', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Legend
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    
    # Add statistics text box
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
    
    # Tight layout
    plt.tight_layout()
    
    return fig