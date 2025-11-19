#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_directory>"
    echo "Recursively finds all *.cdf files and converts them to .tsv next to them."
    exit 1
fi

input_dir="${1%/}"  # remove trailing slash

binary="./cdf_extractor"

if [ ! -x "$binary" ]; then
    echo "Error: $binary not found or not executable."
    exit 1
fi

find "$input_dir" -type f -name "*.CDF" -print0 | while IFS= read -r -d '' cdf_file; do
    echo "Processing: $cdf_file"
    "$binary" "$cdf_file"
done

echo "Batch conversion complete."