# Mass Spec Comparison Workflow

A modular Python tool to compare two mass spectrometry datasets (Reference and Alternative). It identifies the closest matches, calculates differences (Absolute or PPM), sorts the results, and visualizes the distribution of differences.

## Features

*   **Tabular Sorted Search**: Efficiently finds closest matches between datasets using binary search (via `polars`).
*   **Difference Calculation**: Computes absolute differences between specified columns.
*   **PPM Support**: Optionally calculates differences in Parts Per Million (PPM).
*   **Automatic Sorting**: Results are automatically sorted by the difference column (ascending).
*   **Visualization**:
    *   **Absolute Difference**: Histogram with colored bars and legend (max 7 bins).
    *   **PPM Difference**: Density plot with a configurable marker line.
*   **Flexible Input**: Supports CSV, TSV, and other delimited files with auto-detection or explicit delimiters.

## Requirements

*   Python 3.x
*   `polars`
*   `matplotlib`
*   `scipy`
*   `numpy`

## Installation

Ensure you have the required packages installed:

```bash
pip install polars matplotlib scipy numpy
```

## Usage

### Command Line Interface (CLI)

```bash
python mass_spec_compare.py --ref REF_FILE --alt ALT_FILE --ref-col REF_COL --alt-col ALT_COL --output OUTPUT_BASE [OPTIONS]
```

#### Arguments

*   `--ref`: Path to the reference data file (Required).
*   `--alt`: Path to the alternative data file (Required).
*   `--ref-col`: Column name in the reference data to compare (Required).
*   `--alt-col`: Column name in the alternative data to compare (Required).
*   `--output`: Base name for the output files (Required).
*   `--ref-sep`: Delimiter for the reference file (Default: auto-detect or tab).
*   `--alt-sep`: Delimiter for the alternative file (Default: auto-detect or comma).
*   `--ppm`: Flag to calculate and plot differences in PPM.
*   `--ppm-marker`: Marker value for the PPM density plot (Default: 15.0).
*   `--img-fmt`: Image format for the plot (Default: png).

#### Examples

**Basic Comparison (Absolute Difference):**

```bash
python mass_spec_compare.py --ref reference.tsv --alt experimental.csv --ref-col "mz" --alt-col "mz" --output results
```

**PPM Comparison with Custom Marker:**

```bash
python mass_spec_compare.py --ref reference.tsv --alt experimental.csv --ref-col "mz" --alt-col "mz" --output results_ppm --ppm --ppm-marker 20
```

### Python API

You can also import the functions into your own Python scripts or Jupyter notebooks:

```python
import polars as pl
from mass_spec_compare import tabular_sorted_search, calculate_differences, sort_results, plot_differences

# Load data
ref_data = pl.read_csv("reference.tsv", separator="\t")
alt_data = pl.read_csv("experimental.csv", separator=",")

# Perform search
matched_data = tabular_sorted_search(alt_data, "mz", ref_data, "mz")

# Calculate differences (with PPM)
final_data = calculate_differences(matched_data, "mz", "mz", calculate_ppm=True)

# Sort results
final_data = sort_results(final_data, "diff_in_ppm")

# Save and Plot
final_data.write_csv("results.tsv", separator="\t")
plot_differences(final_data, "results", column_to_plot="diff_in_ppm", ppm_marker=20.0)
```

## Output

The tool generates two files:

1.  **`{OUTPUT_BASE}_table.tsv`**: A tab-separated file containing the merged data and calculated differences, sorted by the difference.
2.  **`{OUTPUT_BASE}_plot.png`**: A visualization of the differences (Histogram or Density Plot).
