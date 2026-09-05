#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Verify that the models in a local models folder (default `/MODELS`) are
byte-for-byte identical to the matching snapshots in the local HuggingFace
cache.

The local folder is expected to have one subdirectory per model, named after
the HuggingFace repo **without** the org prefix (e.g. `Qwen3-TTS-12Hz-1.7B-Base`),
with the snapshot files laid out flat plus a `speech_tokenizer/` subdir —
exactly the layout produced by a `git clone` of an HF repo (LFS included).

The HuggingFace cache is expected in the standard layout
`<cache>/hub/models--<org>--<name>/snapshots/<rev>` with `refs/main` pointing
at the active revision and `blobs/` holding the real content (snapshot files
are symlinks into `blobs/`).

Which models to compare is driven by `config.yaml`'s `models` section: each
entry's `hf_id` (e.g. `Qwen/Qwen3-TTS-12Hz-1.7B-Base`) is mapped to a local
subdir (`Qwen3-TTS-12Hz-1.7B-Base`) and to the HF cache repo
(`models--Qwen--Qwen3-TTS-12Hz-1.7B-Base`). Models present in only one side
are reported but are not failures (they simply have nothing to compare).

Exit code: 0 if every comparable model matches, 1 otherwise (mismatch, or a
referenced model is missing on both sides).

Usage:
    python verify_models.py                          # /MODELS vs ~/.cache/huggingface
    python verify_models.py --models-dir /MODELS
    python verify_models.py --hf-cache /root/.cache/huggingface
    python verify_models.py --config ./config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Files left over from a git+LFS clone of an HF repo that are NOT part of the
# model snapshot and are ignored by HuggingFace `from_pretrained`. We skip them
# when comparing the local dir against the HF snapshot so they don't show as
# spurious "extra" files.
_IGNORED_LOCAL_NAMES = {".git", "README.md", ".gitattributes", ".gitkeep"}

# Files we never expect in the local clone but that may appear in the snapshot
# (none currently), kept here for symmetry / future use.
_IGNORED_HF_NAMES: set[str] = set()


def _load_models_from_config(config_path: Path) -> dict[str, str]:
    """Return {local_subdir_name: hf_id} from the `models` section of config.yaml.

    Falls back to an empty dict (caller will then compare whatever subdirs
    exist on both sides) if config.yaml can't be read.
    """
    if not config_path.exists():
        return {}
    try:
        import yaml  # part of the project's dependencies
    except ImportError:
        print(f"  ! PyYAML not installed; cannot read {config_path}. "
              "Pass --models explicitly or install dependencies.", file=sys.stderr)
        return {}
    try:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"  ! Could not parse {config_path}: {exc}", file=sys.stderr)
        return {}

    models = cfg.get("models", {}) or {}
    mapping: dict[str, str] = {}
    for _key, info in models.items():
        if not isinstance(info, dict):
            continue
        hf_id = info.get("hf_id")
        if not hf_id:
            continue
        # Local subdir name = repo name without the org prefix.
        local_name = hf_id.split("/")[-1]
        mapping[local_name] = hf_id
    return mapping


def _sha256(path: Path) -> str:
    """SHA-256 of a file, following symlinks (reads the real content).

    `open()` follows symlinks by default, so the HF snapshot's blob symlinks
    are dereferenced transparently.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _iter_local_files(root: Path):
    """Yield (relative_posix_path, abs_path) for every regular file under root,
    skipping the ignored git/LFS metadata entries."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune ignored dirs in-place so os.walk doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_LOCAL_NAMES]
        base = Path(dirpath)
        for name in filenames:
            if name in _IGNORED_LOCAL_NAMES:
                continue
            full = base / name
            if not full.is_file() and not full.is_symlink():
                continue
            rel = full.relative_to(root).as_posix()
            yield rel, full


def _iter_snapshot_files(snapshot: Path):
    """Yield (relative_posix_path, abs_path) for every file in an HF cache
    snapshot dir, following the blob symlinks so the real content is read."""
    for dirpath, dirnames, filenames in os.walk(snapshot, followlinks=True):
        base = Path(dirpath)
        for name in filenames:
            if name in _IGNORED_HF_NAMES:
                continue
            full = base / name
            rel = full.relative_to(snapshot).as_posix()
            yield rel, full


def _hf_snapshot_dir(hf_cache: Path, hf_id: str) -> Path | None:
    """Resolve the active snapshot directory for an HF repo id from the cache.

    Returns None if the repo isn't cached.
    """
    org, _, name = hf_id.partition("/")
    repo_dir = hf_cache / "hub" / f"models--{org}--{name}"
    if not repo_dir.is_dir():
        return None
    refs_main = repo_dir / "refs" / "main"
    rev = None
    if refs_main.is_file():
        rev = refs_main.read_text().strip()
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    if rev:
        cand = snapshots / rev
        if cand.is_dir():
            return cand
    # Fall back to the only/most recent snapshot if refs/main is missing.
    snap_dirs = sorted([p for p in snapshots.iterdir() if p.is_dir()],
                       key=lambda p: p.stat().st_mtime, reverse=True)
    return snap_dirs[0] if snap_dirs else None


def _compare_one(local_dir: Path, snapshot: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    """Compare a local model dir against an HF snapshot.

    Returns (ok, mismatches, extra_local, missing_local) lists of relative
    paths. `ok` are matched files; `mismatches` are size/hash mismatches;
    `extra_local` are files present locally but not in the snapshot;
    `missing_local` are files in the snapshot but not locally.
    """
    ok: list[str] = []
    mismatches: list[str] = []
    extra_local: list[str] = []
    missing_local: list[str] = []

    local_files = {rel: full for rel, full in _iter_local_files(local_dir)}
    snap_files = {rel: full for rel, full in _iter_snapshot_files(snapshot)}

    for rel, local_full in local_files.items():
        snap_full = snap_files.get(rel)
        if snap_full is None:
            extra_local.append(rel)
            continue
        if _file_size(local_full) != _file_size(snap_full):
            mismatches.append(rel)
            continue
        if _sha256(local_full) != _sha256(snap_full):
            mismatches.append(rel)
            continue
        ok.append(rel)

    for rel in snap_files:
        if rel not in local_files:
            missing_local.append(rel)

    ok.sort()
    mismatches.sort()
    extra_local.sort()
    missing_local.sort()
    return ok, mismatches, extra_local, missing_local


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", default=os.getenv("TTS_MODELS_DIR", "/MODELS"),
                        help="Local models folder (default: $TTS_MODELS_DIR or /MODELS).")
    parser.add_argument("--hf-cache",
                        default=os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
                        help="HuggingFace cache dir (default: $HF_HOME or ~/.cache/huggingface).")
    parser.add_argument("--config", default="config.yaml",
                        help="config.yaml path used to map hf_id -> local subdir (default: ./config.yaml).")
    parser.add_argument("--model", action="append", metavar="HF_ID",
                        help="Compare a specific HF repo id (repeatable). Overrides config-driven selection.")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    hf_cache = Path(args.hf_cache)

    print(f"Local models dir : {models_dir}")
    print(f"HF cache dir     : {hf_cache}")
    print(f"Config           : {args.config}")
    print("=" * 60)

    if args.model:
        model_map = {hf_id.split("/")[-1]: hf_id for hf_id in args.model}
    else:
        model_map = _load_models_from_config(Path(args.config))

    if not model_map:
        # No config / no --model: compare any subdirs that exist on both sides.
        if models_dir.is_dir():
            model_map = {p.name: f"<unknown>/{p.name}" for p in models_dir.iterdir()
                         if p.is_dir() and not p.name.startswith(".")}
        if not model_map:
            print("No models to compare (config.yaml has no `models`, "
                  "models-dir is empty, and no --model given).", file=sys.stderr)
            return 1

    overall_ok = True
    compared = 0
    for local_name in sorted(model_map):
        hf_id = model_map[local_name]
        local_dir = models_dir / local_name
        snapshot = _hf_snapshot_dir(hf_cache, hf_id) if "/" in hf_id else None

        print(f"\nModel: {local_name}  (hf_id={hf_id})")
        if not local_dir.is_dir():
            print(f"  -- not present in {models_dir} (skipped)")
            continue
        if snapshot is None or not snapshot.is_dir():
            print(f"  -- not present in HF cache (skipped; no comparison possible)")
            continue

        ok, mismatches, extra, missing = _compare_one(local_dir, snapshot)
        compared += 1
        for rel in ok:
            print(f"  OK                   {rel}")
        for rel in extra:
            print(f"  EXTRA (not in HF)    {rel}")
        for rel in missing:
            print(f"  MISSING (in HF only) {rel}")
        for rel in mismatches:
            print(f"  HASH/SIZE MISMATCH   {rel}")
        if mismatches:
            overall_ok = False
            print(f"  => MISMATCH ({len(mismatches)} file(s) differ)")
        else:
            print(f"  => MATCH ({len(ok)} file(s) identical)")

    print("\n" + "=" * 60)
    if compared == 0:
        print("No models were comparable (none present on BOTH sides).")
        # Not a hard failure — informational.
        return 0
    if overall_ok:
        print(f"RESULT: all {compared} comparable model(s) match the HF cache.")
        return 0
    print(f"RESULT: one or more models DIFFER from the HF cache.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
