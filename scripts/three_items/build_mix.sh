#!/usr/bin/env bash
# Step 2c: merge the three-butter demos and the LIBERO basket subset into one dataset.
# Usage: bash scripts/three_items/build_mix.sh [--overwrite]
set -euo pipefail
source "$(dirname "$0")/env.sh"

demos=data/three_items_demos
subset=data/libero_basket_subset
mix=data/three_items_mix
for dir in "$demos" "$subset"; do
  if [[ ! -f "$dir/meta/info.json" ]]; then
    echo "Missing $dir. Run collect_demos.sh and prepare_libero_subset.sh first." >&2
    exit 1
  fi
done
if [[ -e "$mix" ]]; then
  if [[ "${1:-}" != "--overwrite" ]]; then
    echo "$mix exists; pass --overwrite to rebuild it." >&2
    exit 1
  fi
  rm -rf "$mix"
fi

.venv/bin/lerobot-edit-dataset \
  --new_repo_id=local/three_items_mix \
  --new_root="$mix" \
  --operation.type=merge \
  --operation.repo_ids="['local/three_items_demos', 'local/libero_basket_subset']" \
  --operation.roots="['$demos', '$subset']"

.venv/bin/python - "$mix" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd
from lerobot_env_libero_three_items.task import LANGUAGE, LANGUAGE_VARIANTS

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
episodes = pd.concat(pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet")))
episodes["task"] = episodes["tasks"].str[0]
butter = episodes["task"].isin([LANGUAGE, *LANGUAGE_VARIANTS])
print(f"{root}: {info['total_episodes']} episodes, {info['total_frames']} frames")
for label, mask in (("three-butter demos", butter), ("LIBERO basket subset", ~butter)):
    frames = int(episodes.loc[mask, "length"].sum())
    print(f"  {label}: {int(mask.sum())} episodes, {frames} frames ({frames / info['total_frames']:.0%})")
PY
