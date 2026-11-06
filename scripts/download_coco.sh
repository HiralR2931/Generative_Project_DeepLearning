#!/usr/bin/env bash
# Downloads COCO 2017 validation images (~1 GB) and caption annotations (~250 MB).
# Run from the repo root:  bash scripts/download_coco.sh
set -euo pipefail
mkdir -p data/raw
cd data/raw
wget -nc http://images.cocodataset.org/zips/val2017.zip
wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q -n val2017.zip
unzip -q -n annotations_trainval2017.zip
echo "Done. Images: data/raw/val2017/  Captions: data/raw/annotations/captions_val2017.json"
