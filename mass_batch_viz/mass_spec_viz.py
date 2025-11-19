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

def process_file(args):
    file_path, mz_low, mz_high, source, file_format = args
    try:
        sep = '\t' if file_format == 'tsv' else ','
        df = (pl.read_csv(file_path, separator=sep, infer_schema_length=10000)
              .filter(pl.col("mz").is_between(mz_low, mz_high))
              .select([
                  "retention_time",
                  "mz",
                  "intensity",
                  pl.lit(source).alias("source")
              ]))
        return df
    except Exception:
        return pl.DataFrame(schema={"retention_time": pl.Float64, "mz": pl.Float64, "intensity": pl.Float64, "source": pl.Utf8})

def main():
    parser = argparse.ArgumentParser(description="Visualize mass spec data chromatogram for given m/z and ppm tolerance.")
    parser.add_argument("path", help="Path to file or directory containing TSV/CSV files.")
    parser.add_argument("mz", type=float, help="Target m/z value.")
    parser.add_argument("ppm", type=float, nargs="?", default=15.0, help="PPM tolerance (default: 15).")
    parser.add_argument("--level", choices=["file", "dir"], default="dir", help="Grouping level: 'file' or 'dir' (default: dir).")
    parser.add_argument("--plot_type", choices=["scatter", "line", "both"], default="scatter", help="Plot type: 'scatter', 'line', or 'both' (default: scatter).")
    parser.add_argument("--format", choices=["tsv", "csv"], default="tsv", help="File format (default: tsv).")
    parser.add_argument("--export", nargs='?', const='auto', default=None, help="Export plot to SVG file. If no filename, uses timestamp.")
    parser.add_argument("--export-format", choices=["svg", "png"], default="svg", help="Export image format (default: svg).")

    args = parser.parse_args()

    mz_low = args.mz * (1 - args.ppm / 1_000_000)
    mz_high = args.mz * (1 + args.ppm / 1_000_000)

    if os.path.isfile(args.path) and args.path.endswith(f".{args.format}"):
        files = [args.path]
    else:
        pattern = f"**/*.{args.format}"
        files = glob.glob(os.path.join(args.path, pattern), recursive=True)

    process_args = []
    for file_path in files:
        if args.level == "file":
            source = os.path.basename(file_path)
        else:
            source = os.path.basename(os.path.dirname(file_path))
        process_args.append((file_path, mz_low, mz_high, source, args.format))

    if not process_args:
        print("No matching files found.")
        return

    with mp.Pool(processes=mp.cpu_count()) as pool:
        dfs = pool.map(process_file, process_args)

    df_all = pl.concat(dfs, how="vertical")
    if df_all.is_empty():
        print(f"No data points found within m/z [{mz_low:.4f}, {mz_high:.4f}].")
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

        if "scatter" in args.plot_type or args.plot_type == "both":
            ax.scatter(rt, inten, c=[col], label=src if show_legend else None, alpha=0.8, s=4)
        if "line" in args.plot_type or args.plot_type == "both":
            ax.plot(rt, inten, c=col, label=src if show_legend else None, linewidth=2, alpha=0.9)

    ax.set_xlabel("Retention Time")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Extracted Ion Chromatogram: m/z {args.mz:.2f} ± {args.ppm} ppm\n"
                 f"Level: {args.level}, Plot: {args.plot_type}, Sources: {n_sources}")
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if args.export is not None:
        if args.export == 'auto':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outpath = f"eic_mz{args.mz:.2f}_ppm{args.ppm:.0f}_level{args.level}_{timestamp}.{args.export_format}"
        else:
            outpath = args.export
            if not outpath.lower().endswith(('.svg', '.png')):
                outpath += f'.{args.export_format}'
        fig.savefig(outpath, format=args.export_format.lower(), bbox_inches='tight', dpi=300)
        print(f"Plot exported to {outpath}")
    plt.show()

if __name__ == "__main__":
    main()