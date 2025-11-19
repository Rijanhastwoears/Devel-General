#!/usr/bin/env python3
import argparse
import glob
import os
import multiprocessing as mp
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from mass_spec_viz import process_file

def plot_chromatogram(df_all, mz, ppm, level="dir", plot_type="scatter", export=None, export_format="png"):
    if df_all.is_empty():
        print(f"No data points found for m/z {mz:.4f} ± {ppm} ppm.")
        return
    df_pd = df_all.to_pandas()
    unique_sources = df_pd["source"].unique()
    n_sources = len(unique_sources)
    show_legend = n_sources <= 15
    dark_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    color_map = {src: dark_colors[i % len(dark_colors)] for i, src in enumerate(unique_sources)}
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, color="gray", alpha=0.3, linestyle="--")
    for src in unique_sources:
        group = df_pd[df_pd["source"] == src].sort_values("retention_time")
        rt = group["retention_time"].values
        inten = group["intensity"].values
        col = color_map[src]
        if "scatter" in plot_type or plot_type == "both":
            ax.scatter(rt, inten, c=[col], label=src if show_legend else None, alpha=0.8, s=4)
        if "line" in plot_type or plot_type == "both":
            ax.plot(rt, inten, c=col, label=src if show_legend else None, linewidth=2, alpha=0.9)
    ax.set_xlabel("Retention Time")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Extracted Ion Chromatogram: m/z {mz:.2f} ± {ppm} ppm\\n"
                 f"Level: {level}, Plot: {plot_type}, Sources: {n_sources}")
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    saved = False
    if export == "auto":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = f"batch_eic_mz{mz:.2f}_ppm{ppm:.0f}_level{level}_{timestamp}.{export_format}"
        fig.savefig(outpath, format=export_format.lower(), bbox_inches='tight', dpi=300)
        print(f"Plot exported to {outpath}")
        saved = True
    if not saved:
        plt.show()

def get_files(path, file_format):
    if os.path.isfile(path) and path.endswith(f".{file_format}"):
        return [path]
    pattern = f"**/*.{file_format}"
    return glob.glob(os.path.join(path, pattern), recursive=True)

def process_mz(path, mz, ppm, level="dir", file_format="tsv"):
    mz_low = mz * (1 - ppm / 1_000_000)
    mz_high = mz * (1 + ppm / 1_000_000)
    files = get_files(path, file_format)
    if not files:
        print(f"No matching files found in {path}.")
        return pl.DataFrame(schema={"retention_time": pl.Float64, "mz": pl.Float64, "intensity": pl.Float64, "source": pl.Utf8})
    process_args = []
    for file_path in files:
        source = os.path.basename(file_path) if level == "file" else os.path.basename(os.path.dirname(file_path))
        process_args.append((file_path, mz_low, mz_high, source, file_format))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        dfs = pool.map(process_file, process_args)
    df_all = pl.concat(dfs, how="vertical")
    return df_all

def main():
    parser = argparse.ArgumentParser(description="Batch visualization of mass spec data chromatograms for list of m/z or CSV table.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", help="Path to CSV/TSV file with 'mz' (req float), optional 'path' (uses --path if missing), optional 'ppm' (defaults 15).")
    input_group.add_argument("--mzs", nargs="+", type=float, help="List of target m/z values.")
    parser.add_argument("--path", help="Single path to dir/file (req with --mzs; default for --csv if no 'path' col).")
    parser.add_argument("--ppm", type=float, default=15.0, help="PPM tolerance for --mzs mode (default 15).")
    parser.add_argument("--level", choices=["file", "dir"], default="dir", help="Grouping level (default: dir).")
    parser.add_argument("--plot-type", choices=["scatter", "line", "both"], default="scatter", help="Plot type (default: scatter).")
    parser.add_argument("--format", choices=["tsv", "csv"], default="tsv", help="File format (default: tsv).")
    parser.add_argument("--interactive", action="store_true", help="Show plots interactively (default: auto-export SVG).")
    parser.add_argument("--export-format", choices=["svg", "png"], default="png", help="Auto-export image format (default: png).")
    args = parser.parse_args()

    if args.csv:
        # Read input CSV/TSV robustly
        schema_sample = pl.read_csv(args.csv, n_rows=1)
        columns = schema_sample.columns
        if "mz" not in columns:
            parser.error("CSV must have 'mz' column.")
        select_cols = ["mz"]
        if "path" in columns:
            select_cols.append("path")
        if "ppm" in columns:
            select_cols.append("ppm")
        df_input = pl.read_csv(args.csv).select(select_cols)
        # PPM default
        if "ppm" not in df_input.columns:
            df_input = df_input.with_columns(pl.lit(15.0).alias("ppm"))
        else:
            df_input = df_input.with_columns(pl.col("ppm").fill_null(15.0))
        # Path default
        if "path" not in df_input.columns:
            if args.path:
                df_input = df_input.with_columns(pl.lit(args.path).alias("path"))
            else:
                parser.error("CSV lacks 'path' column and no --path provided.")
        for row_dict in df_input.iter_rows(named=True):
            mz = row_dict["mz"]
            path = row_dict["path"]
            ppm = row_dict["ppm"]
            df_all = process_mz(path, mz, ppm, args.level, args.format)
            plot_chromatogram(df_all, mz, ppm, args.level, args.plot_type, None if args.interactive else "auto", args.export_format)
    elif args.mzs:
        if not args.path:
            parser.error("--path is required when using --mzs.")
        ppm = args.ppm
        for mz in args.mzs:
            df_all = process_mz(args.path, mz, ppm, args.level, args.format)
            plot_chromatogram(df_all, mz, ppm, args.level, args.plot_type, None if args.interactive else "auto", args.export_format)

if __name__ == "__main__":
    main()