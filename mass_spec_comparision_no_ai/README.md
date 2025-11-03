# mass_spec_comparision_no_ai

Utilities to compare alternative mass spectrometry feature lists with a ground truth list
using nearest-neighbour matching on m/z and retention time (RT).

## Files
- `mass_spec_comparision.py` - matching, difference calculation, and plotting helpers.

## match_ground_truth
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

Example:
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