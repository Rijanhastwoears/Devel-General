# Multi-dimensional Mass Spec Comparison

## Overview

`mass_spec_multidim_compare.py` is a Python tool designed to compare two mass spectrometry datasets (Reference and Alternative) across multiple dimensions simultaneously. Unlike traditional 1D comparisons (e.g., just m/z), this tool uses **K-d trees** to find the nearest neighbors in a multi-dimensional space (e.g., m/z AND Retention Time), providing a more accurate matching process for complex data.

## Use Cases

### 1. Matching Features by Mass and Retention Time
The most common use case is matching experimental features to a reference library based on both mass-to-charge ratio (m/z) and retention time (RT).

**Example:**
You have an experimental peak list (`experimental.csv`) and a reference library (`library.tsv`). You want to find the library entry that is closest to each experimental peak, considering that a peak might match by mass but be at the wrong time.

```bash
python mass_spec_multidim_compare.py \
  --ref library.tsv \
  --alt experimental.csv \
  --dim mz,Library_MZ,Exp_MZ \
  --dim rt,Library_RT,Exp_RT \
  --output results
```

### 2. Cross-Platform Comparison
Comparing results from two different instruments or processing pipelines where the feature coordinates (m/z, RT, Ion Mobility, etc.) might have systematic shifts or different variances.

### 3. Interactive Python Usage
You can use the tool programmatically in your own scripts or Jupyter notebooks.

```python
import polars as pl
from mass_spec_multidim_compare import run_multidim_comparison

# Load data
ref_data = pl.read_csv("reference.tsv", separator="\t")
alt_data = pl.read_csv("experimental.csv", separator=",")

# Define dimensions: (name, ref_col, alt_col)
dims = [
    ("mz", "mz", "mz"),
    ("rt", "rt", "rt")
]

# Run comparison
result = run_multidim_comparison(ref_data, alt_data, dims, "output_base")
```

## Dependencies

*   **Python 3.x**
*   **Polars**: For fast data manipulation.
*   **NumPy**: For numerical operations.
*   **SciPy**: For the K-d tree implementation (`scipy.spatial.KDTree`).
*   **Matplotlib**: For generating the correspondence plot.

Install via pip:
```bash
pip install polars numpy scipy matplotlib
```

## Internals: The How and Why

### The Challenge: Multi-dimensional Distance
In mass spectrometry, dimensions often have vastly different units and scales.
*   **m/z**: Typically ranges from 100-2000, with precision needed at 0.001 level.
*   **Retention Time**: Ranges from 0-100 minutes, with precision at 0.1 level.

If you simply calculated the Euclidean distance, the dimension with the larger numerical values (or larger variance) would dominate the distance metric. A difference of 1 minute in RT is "farther" than a difference of 0.1 Da in m/z, even if 0.1 Da is a huge error in mass spec terms.

### The Solution: Z-Score Normalization + K-d Trees

1.  **Z-Score Normalization**:
    Before comparison, the tool normalizes each dimension using **Z-scores** (Standardization).
    $$ z = \frac{x - \mu}{\sigma} $$
    *   **$\mu$ (Mean)** and **$\sigma$ (Standard Deviation)** are calculated from the **Reference** dataset.
    *   These statistics are then applied to the **Alternative** dataset.
    *   **Why?** This scales all dimensions to units of "standard deviations." A difference of 1 sigma in RT is now treated as equivalent to a difference of 1 sigma in m/z. This allows for a fair, "unit-less" distance calculation that respects the natural variability of the data.

2.  **K-d Tree (k-dimensional tree)**:
    *   The tool builds a K-d tree from the normalized Reference data.
    *   It queries this tree with the normalized Alternative data to find the nearest neighbor.
    *   **Why?** K-d trees provide an efficient ($O(\log N)$) way to find nearest neighbors in multi-dimensional space, avoiding the computational cost of comparing every point to every other point ($O(N^2)$).

3.  **Correspondence Plot**:
    *   Visualizes the matches in the first two dimensions.
    *   Reference points are shown as **Green Circles**.
    *   Alternative points are shown as **Blue Triangles**.
    *   Lines connect the matched pairs, allowing for quick visual verification of the matching quality and detection of systematic shifts.

## Output Files

The tool generates three files:

1.  **`{OUTPUT}_table.tsv`**: The main results table containing the merged data, calculated differences, and normalized distances.
2.  **`{OUTPUT}_plot.png`**: A correspondence scatter plot of the first two dimensions.
3.  **`{OUTPUT}_stats.tsv`**: A summary statistics file containing:
    *   **Total Euclidean Distance**: Sum of normalized distances for all matches.
    *   **Mean Euclidean Distance**: Average normalized distance.
    *   **Median Euclidean Distance**: Median normalized distance.
