# Mass Spectrometry Data Comparison

This repository provides Python functions for comparing alternative mass spectrometry (MS) data against a ground truth dataset. The primary goal is to assess the accuracy of `m/z` (mass-to-charge ratio) and `RT` (retention time) measurements by matching and calculating differences, and then visualizing these differences with tolerance-based scatter plots.

## `mass_spec_comparision.py`

This file contains the core logic for matching, calculating differences, and plotting.

### Functions

#### `match_ground_truth(alt_data: pl.DataFrame, ground_truth: pl.DataFrame) -> pl.DataFrame`

Matches alternative mass spectrometry data against ground truth data.

For each row in `alt_data`, this function finds the nearest neighbor in `ground_truth` based on two separate criteria:
1.  **m/z value**: It finds the ground truth entry with the closest m/z. The m/z and RT of this matched ground truth entry are stored in new columns `mz_by_mz` and `rt_by_mz`.
2.  **RT value**: It finds the ground truth entry with the closest retention time (RT). The m/z and RT of this matched ground truth entry are stored in new columns `mz_by_rt` and `rt_by_rt`.

**Parameters:**
-   `alt_data` : `pl.DataFrame`
    A Polars DataFrame containing the alternative data. Must have 'mz' and 'rt' columns.
-   `ground_truth` : `pl.DataFrame`
    A Polars DataFrame containing the ground truth data. Must have 'mz' and 'rt' columns.

**Returns:**
-   `pl.DataFrame`
    The original `alt_data` DataFrame with four new columns appended: `mz_by_mz`, `rt_by_mz`, `mz_by_rt`, and `rt_by_rt`, containing the matched ground truth values.

#### `calculate_differences(matched_data: pl.DataFrame) -> pl.DataFrame`

Calculates the parts per million (ppm) and retention time (RT) differences between the alternative mass spectrometry data and its matched ground truth data.

This function computes four new columns based on the matching criteria:
-   `ppm_diff_by_mz`: The ppm difference between the 'mz' of the alternative data and the 'mz_by_mz' (ground truth mz matched by mz).
-   `rt_diff_by_mz`: The absolute RT difference between the 'rt' of the alternative data and the 'rt_by_mz' (ground truth rt matched by mz).
-   `ppm_diff_by_rt`: The ppm difference between the 'mz' of the alternative data and the 'mz_by_rt' (ground truth mz matched by rt).
-   `rt_diff_by_rt`: The absolute RT difference between the 'rt' of the alternative data and the 'rt_by_rt' (ground truth rt matched by rt).

The ppm difference is calculated using the formula: `((abs(measured - theoretical) / theoretical) * 1,000,000)`.

**Parameters:**
-   `matched_data` : `pl.DataFrame`
    A Polars DataFrame that is the output of the `match_ground_truth` function. It must contain the columns: 'mz', 'rt', 'mz_by_mz', 'rt_by_mz', 'mz_by_rt', 'rt_by_rt'.

**Returns:**
-   `pl.DataFrame`
    Input dataframe with additional difference columns.

#### `plot_tolerance_scatter(x_series: pl.Series, y_series: pl.Series, x_tol: float, y_tol: float, x_label: str = None, y_label: str = None, title: str = None, figsize: tuple = (10, 8)) -> plt.Figure`

Create an academic-style scatter plot with tolerance-based color coding.

**Color scheme:**
-   **Green**: Below both `x_tol` and `y_tol` (good matches)
-   **Blue**: Below `x_tol` only
-   **Yellow**: Below `y_tol` only
-   **Red**: Above both tolerances (poor matches)

**Parameters:**
-   `x_series` : `pl.Series`
    Data for x-axis.
-   `y_series` : `pl.Series`
    Data for y-axis.
-   `x_tol` : `float`
    Tolerance threshold for x-axis values.
-   `y_tol` : `float`
    Tolerance threshold for y-axis values.
-   `x_label` : `str`, optional
    Label for x-axis (defaults to series name).
-   `y_label` : `str`, optional
    Label for y-axis (defaults to series name).
-   `title` : `str`, optional
    Plot title.
-   `figsize` : `tuple`, optional
    Figure size (width, height).

**Returns:**
-   `plt.Figure`
    The matplotlib figure object.

### Usage Example

```python
import polars as pl
import matplotlib.pyplot as plt
from mass_spec_comparision import match_ground_truth, calculate_differences, plot_tolerance_scatter

# Sample Data
alt_data = pl.DataFrame({
    "mz": [100.001, 200.005, 300.010, 400.020, 500.030],
    "rt": [1.0, 2.1, 3.0, 4.2, 5.0]
})

ground_truth = pl.DataFrame({
    "mz": [100.000, 200.000, 300.000, 400.000, 500.000, 100.002, 200.007],
    "rt": [1.0, 2.0, 3.0, 4.0, 5.0, 1.1, 2.2]
})

# 1. Match ground truth
matched_df = match_ground_truth(alt_data, ground_truth)
print("Matched Data:")
print(matched_df)

# 2. Calculate differences
diff_df = calculate_differences(matched_df)
print("\nDifferences Data:")
print(diff_df)

# 3. Plot tolerance scatter
fig = plot_tolerance_scatter(
    x_series=diff_df["ppm_diff_by_mz"],
    y_series=diff_df["rt_diff_by_mz"],
    x_tol=5.0,  # 5 ppm tolerance
    y_tol=0.1,  # 0.1 minute RT tolerance
    x_label="m/z difference (ppm)",
    y_label="RT difference (minutes)",
    title="m/z and RT Differences (Matched by m/z)"
)
plt.show()

fig_rt = plot_tolerance_scatter(
    x_series=diff_df["ppm_diff_by_rt"],
    y_series=diff_df["rt_diff_by_rt"],
    x_tol=5.0,  # 5 ppm tolerance
    y_tol=0.1,  # 0.1 minute RT tolerance
    x_label="m/z difference (ppm)",
    y_label="RT difference (minutes)",
    title="m/z and RT Differences (Matched by RT)"
)
plt.show()
```

## Installation

Make sure you have the necessary libraries installed:

```bash
pip install polars matplotlib numpy
```

## Contributing

Feel free to open issues or pull requests if you have suggestions or improvements.
