# mass_spec_comparision_no_ai

Utilities to compare alternative mass spectrometry feature lists with a ground truth list
using nearest-neighbour matching on m/z and retention time (RT).

## Files
- `mass_spec_comparision.py` - matching, difference calculation, plotting helpers, and CLI interface.

## Installation

```bash
# Install required dependencies
pip install polars matplotlib numpy
```

## Command Line Interface

The script provides a command-line interface for comparing mass spectrometry datasets:

```bash
python mass_spec_comparision.py [OPTIONS] alt_data ground_truth
```

### Positional Arguments
- `alt_data`: Path to alternative data file (CSV, Parquet, Excel)
- `ground_truth`: Path to ground truth data file (CSV, Parquet, Excel)

### Options
- `--mz-tolerance FLOAT`: Absolute tolerance for m/z matching (Da)
- `--rt-tolerance FLOAT`: Absolute tolerance for retention time matching (min)
- `--output, -o TEXT`: Output file path for results (default: stdout)
- `--output-dir, -d TEXT`: Output directory for results and plots (default: current directory)
- `--formats TEXT [TEXT ...]`: Output formats to generate (choices: csv, parquet, excel)
- `--no-plots`: Skip plot generation
- `--plots-dir TEXT`: Directory for plots (default: plots)
- `--quiet, -q`: Suppress informational output
- `-h, --help`: Show help message and exit

### Examples

```bash
# Basic usage
python mass_spec_comparision.py alt_data.csv ground_truth.csv

# With tolerances and custom output
python mass_spec_comparision.py alt_data.csv ground_truth.csv --mz-tolerance 0.01 --rt-tolerance 0.5 --output results.csv

# Generate multiple output formats and plots
python mass_spec_comparision.py alt_data.parquet ground_truth.parquet --output-dir results/ --formats csv parquet excel

# Quiet mode with custom plots directory
python mass_spec_comparision.py alt_data.csv ground_truth.csv --quiet --plots-dir comparison_plots
```

### Input Requirements
Input files must contain the following columns:
- `mz`: Mass-to-charge ratio values
- `rt`: Retention time values

### Output
The CLI generates:
1. **Matched data** with additional columns:
   - `mz_by_mz`, `rt_by_mz`: ground-truth mz and rt matched by nearest mz
   - `mz_by_rt`, `rt_by_rt`: ground-truth mz and rt matched by nearest rt
2. **Difference calculations**:
   - `ppm_diff_by_mz`, `ppm_diff_by_rt`: PPM differences
   - `dalton_diff_by_mz`, `dalton_diff_by_rt`: Dalton differences
   - `rt_diff_by_mz`, `rt_diff_by_rt`: Retention time differences
3. **Comparison plots** (if not disabled):
   - Tolerance-based scatter plots with color coding
   - Statistical summaries included in plots

## Python API

### match_ground_truth
Signature:
```python
match_ground_truth(alt_data: pl.DataFrame,
                   ground_truth: pl.DataFrame,
                   *,
                   tolerance_mz: float | None = None,
                   tolerance_rt: float | None = None) -> pl.DataFrame
```

- Matches each row in `alt_data` to the nearest row in `ground_truth` by `mz` and independently by `rt`.
- Returns `alt_data` with four new columns:
  - `mz_by_mz`, `rt_by_mz`: ground-truth mz and rt matched by nearest mz
  - `mz_by_rt`, `rt_by_rt`: ground-truth mz and rt matched by nearest rt
- If `tolerance_mz` or `tolerance_rt` is provided, matches with absolute difference greater than the tolerance are set to null.

### calculate_differences
Signature:
```python
calculate_differences(matched_data: pl.DataFrame) -> pl.DataFrame
```

Given the output of `match_ground_truth`, computes ppm and RT differences:
- `ppm_diff_by_mz`, `ppm_diff_by_rt`: PPM differences
- `dalton_diff_by_mz`, `dalton_diff_by_rt`: Dalton differences
- `rt_diff_by_mz`, `rt_diff_by_rt`: Retention time differences

### plot_tolerance_scatter
Signature:
```python
plot_tolerance_scatter(x_series: pl.Series, y_series: pl.Series,
                       x_tol: float, y_tol: float,
                       x_label: str = None, y_label: str = None,
                       title: str = None, figsize: tuple = (10, 8)) -> plt.Figure
```

Create academic-style scatter plots with tolerance-based color coding:
- **Green**: Within both x_tol and y_tol (good matches)
- **Blue**: Within x_tol only
- **Yellow**: Within y_tol only
- **Red**: Outside both tolerances (poor matches)

### Example (Python API)
```python
import polars as pl
from mass_spec_comparision import match_ground_truth, calculate_differences

alt = pl.DataFrame({"mz": [100.0, 200.0], "rt": [5.1, 10.2]})
gt = pl.DataFrame({"mz": [99.99, 200.01], "rt": [5.0, 10.3]})

matched = match_ground_truth(alt, gt, tolerance_mz=0.05, tolerance_rt=0.5)
diffs = calculate_differences(matched)
```

## Notes
- The implementation uses Polars `join_asof` (strategy="nearest") under the hood. Inputs are cast to Float64 and the original input order is preserved.
- Use the `tolerance_*` parameters when you want to reject far-away matches rather than accept the nearest match unconditionally.
- CLI supports CSV, Parquet, and Excel file formats for both input and output.
- Generated plots include statistical summaries and use a color scheme that reflects tolerance compliance.