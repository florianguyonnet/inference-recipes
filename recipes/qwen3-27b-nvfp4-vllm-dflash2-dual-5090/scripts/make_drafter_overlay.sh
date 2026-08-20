#!/usr/bin/env bash
set -euo pipefail

# Build a drafter overlay whose position table covers a YaRN-extended target.
#
# The DFlash2 drafter ships max_position_embeddings=262144. Serving a target at
# 512k feeds it positions past that, the cos/sin cache is indexed out of bounds
# and the run dies on `CUDA error: device-side assert triggered` while building
# attention metadata. Only the table size is wrong: the drafter attends in a
# 2048-token sliding window, so its relative geometry — and its acceptance — is
# unchanged by extending it. RoPE type and theta stay exactly as trained.
#
# Usage: ./scripts/make_drafter_overlay.sh [repo_id] [max_position_embeddings]
# Safe to re-run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

HF_HOME="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
REPO_ID="${1:-${DRAFT_REPO_ID:-incoai/Qwen3.8-27B-DFlash2}}"
MAX_POS="${2:-${DRAFTER_MAX_POSITIONS:-524288}}"

python3 - "${HF_HOME}" "${REPO_ID}" "${MAX_POS}" <<'PY'
import glob, json, os, sys

hf_home, repo_id, max_pos = sys.argv[1], sys.argv[2], int(sys.argv[3])
cache = os.path.join(hf_home, "hub", "models--" + repo_id.replace("/", "--"))
ref = os.path.join(cache, "refs", "main")
snap = None
if os.path.isfile(ref):
    snap = os.path.join(cache, "snapshots", open(ref).read().strip())
if not snap or not os.path.isdir(snap):
    cands = sorted(glob.glob(os.path.join(cache, "snapshots", "*")))
    snap = cands[0] if cands else None
if not snap or not os.path.isdir(snap):
    raise SystemExit(f"ERROR: {repo_id} not in {hf_home}. Run scripts/download_model.sh first.")
if not glob.glob(os.path.join(snap, "*.safetensors")):
    raise SystemExit(f"ERROR: no *.safetensors in {snap} (incomplete download?)")

overlay = os.path.join(hf_home, "overlay-drafter-512k")
os.makedirs(overlay, exist_ok=True)
for f in glob.glob(os.path.join(overlay, "*")):
    os.remove(f)

linked = 0
for f in sorted(glob.glob(os.path.join(snap, "*"))):
    base = os.path.basename(f)
    if base == "config.json":
        continue
    os.symlink(os.path.relpath(f, overlay), os.path.join(overlay, base))
    linked += 1

cfg = json.load(open(os.path.join(snap, "config.json")))
before = cfg.get("max_position_embeddings")
cfg["max_position_embeddings"] = max_pos
json.dump(cfg, open(os.path.join(overlay, "config.json"), "w"), indent=2)

print(f"==> Overlay : {overlay}")
print(f"    Snapshot: {snap}")
print(f"==> Done: {linked} files symlinked (relative), "
      f"max_position_embeddings {before} -> {max_pos}, "
      f"rope {json.dumps(cfg.get('rope_parameters'))} untouched.")
PY
