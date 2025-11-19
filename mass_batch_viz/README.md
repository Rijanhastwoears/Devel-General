# Mass Spectrometry Extracted Ion Chromatogram (EIC) Visualizer

## Overview

Two Python scripts for visualizing mass spectrometry data as EIC plots (chromatograms of intensity vs. retention time for specific m/z ranges).

- [`mass_spec_viz.py`](mass_spec_viz.py): Single m/z value plotting from a file or directory of TSV/CSV files.
- [`mass_spec_batch_viz.py`](mass_spec_batch_viz.py): Batch plotting for multiple m/z values. Supports:
  - List of m/z with a single data path (uniform PPM tolerance).
  - CSV table input with per-m/z paths and PPM tolerances (priority over defaults).

Both use multiprocessing for efficiency, Polars for data processing, and Matplotlib for high-quality plots (scatter/line/both). Handles large datasets from directories like `AIAEXPRT_SWG*.AIA/`.

## Requirements

Python 3.8+ with:
```
pip install polars matplotlib numpy
```
Assumes input files (TSV/CSV) have columns: `retention_time` (float), `mz` (float), `intensity` (float).

Make scripts executable:
```bash
chmod +x mass_spec_viz.py mass_spec_batch_viz.py
```

## Single Plot: `mass_spec_viz.py`

Extract and plot data within m/z ± PPM tolerance.

### Basic Usage
```bash
./mass_spec_viz.py <path> <mz> [ppm] [--level file|dir] [--plot-type scatter|line|both] [--format tsv|csv] [--export]
```

- `path`: File or directory (recursive glob `**/*.{format}`).
- `mz`: Target m/z (float).
- `ppm`: PPM tolerance (default: 15.0).
- `--level`: Group by `file` (filename) or `dir` (parent dir, default).
- `--plot-type`: `scatter` (default), `line`, `both`.
- `--format`: `tsv` (default), `csv`.
- `--export`: Save SVG (auto: timestamped filename).

### Examples
```bash
# Plot from directory, group by dir
./mass_spec_viz.py AIAEXPRT_SWG5.AIA/ 445.2 --ppm 20 --level dir --export

# Single file, line plot
./mass_spec_viz.py AIAEXPRT_SWG5.AIA/120420B-08.tsv 300.1 --plot-type line --format tsv
```

**Output**: Interactive plot (or SVG export). Legend if ≤15 sources.

## Batch Plot: `mass_spec_batch_viz.py`

Plots multiple EICs sequentially. Mutually exclusive inputs:
- `--mzs`: List of m/z + single `--path` + uniform `--ppm`.
- `--csv`: Table with `mz` (req), optional `path` (uses CLI `--path` if missing), optional `ppm` (default 15).

Common options: `--level`, `--plot-type`, `--format`, `--interactive` (show interactively; default: auto-export SVG).

### Mode 1: m/z List + Single Path
```bash
./mass_spec_batch_viz.py --mzs 100.1 200.2 300.3 --path AIAEXPRT_SWG5.AIA/ --ppm 15
```

### Mode 2: CSV Table (Per-Row Priority)
**Example `table.csv` (with per-row paths):**
```csv
mz,path,ppm
445.2,AIAEXPRT_SWG5.AIA/,10
500.1,AIAEXPRT_SWG6.AIA/,20
```
```bash
./mass_spec_batch_viz.py --csv table.csv --level dir --plot-type both
```

**Example without `path` column (uses CLI `--path`):**
```csv
mz,ppm
445.2,10
```
```bash
./mass_spec_batch_viz.py --csv table_no_path.csv --path AIAEXPRT_SWG5.AIA/ --level dir
```

## Notes
- PPM range: `mz_low = mz * (1 - ppm/1e6)`, `mz_high = mz * (1 + ppm/1e6)`.
- Colors: Fixed palette for consistency.
- Empty results: Printed warning, no plot.
- Performance: Multiprocessing over all files per m/z.
- Exports: Timestamped SVG (`batch_eic_mz445.20_ppm10_leveldir_YYYYMMDD_HHMMSS.svg`).

For issues, check data columns/formats match requirements.