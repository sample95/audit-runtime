#!/usr/bin/env bash
# Generate the checkpoint integrity manifest. Run from this folder (fast on local disk):
#   cd src_clean/models && bash make_sha256sums.sh
set -e
sha256sum */*.zip */*.pkl > SHA256SUMS
echo "$(wc -l < SHA256SUMS) checksums written to SHA256SUMS"
