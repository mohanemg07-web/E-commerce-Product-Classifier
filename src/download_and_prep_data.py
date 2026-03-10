"""
Smart E-commerce Product Classifier — Data Ingestion & Preprocessing
=====================================================================
Automates the entire data pipeline:
  1. Downloads ``paramaggarwal/fashion-product-images-small`` from Kaggle
  2. Parses ``styles.csv`` to map image IDs → masterCategory labels
  3. Restructures images into PyTorch ImageFolder layout
  4. Prunes minority classes (< 500 images) to prevent class-imbalance issues
  5. Cleans up temporary artifacts

Usage::

    python src/download_and_prep_data.py

Prerequisites:
    • ``pip install kaggle pandas``
    • Kaggle API credentials at ``~/.kaggle/kaggle.json``
      (or ``%USERPROFILE%/.kaggle/kaggle.json`` on Windows)
      → https://www.kaggle.com/docs/api#authentication
"""

import os
import sys
import shutil
import zipfile
import tempfile
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
DATASET_SLUG   = "paramaggarwal/fashion-product-images-small"
MIN_IMAGES     = 500   # Prune categories with fewer images than this


def _check_kaggle_credentials() -> None:
    """
    Verify that the Kaggle API key exists before attempting any download.
    Provides a clear, actionable error message if it is missing.
    """
    # Kaggle looks for credentials in these locations
    kaggle_json = (
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle"))
        .expanduser() / "kaggle.json"
    )
    # Environment-variable auth is also valid
    has_env = ("KAGGLE_USERNAME" in os.environ and
               "KAGGLE_KEY" in os.environ)

    if not kaggle_json.exists() and not has_env:
        print("=" * 60)
        print("  ERROR: Kaggle API credentials not found!")
        print("=" * 60)
        print()
        print("  Option A — credential file:")
        print(f"    Place your kaggle.json at: {kaggle_json}")
        print()
        print("  Option B — environment variables:")
        print("    export KAGGLE_USERNAME=your_username")
        print("    export KAGGLE_KEY=your_api_key")
        print()
        print("  Get your API key at:")
        print("    https://www.kaggle.com/settings → API → Create New Token")
        print("=" * 60)
        sys.exit(1)

    print("✅ Kaggle credentials found")


# ─────────────────────────────────────────────────────────────
# Step 1 — Download & Extract
# ─────────────────────────────────────────────────────────────
def download_and_extract(tmp_dir: Path) -> Path:
    """
    Download the Kaggle dataset archive and extract it.

    Returns:
        Path to the extracted dataset root inside *tmp_dir*.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    print(f"\n📥 Downloading dataset: {DATASET_SLUG}")
    print(f"   Destination: {tmp_dir}")

    api = KaggleApi()
    api.authenticate()

    # Download all files as a zip
    api.dataset_download_files(
        DATASET_SLUG,
        path=str(tmp_dir),
        unzip=False,
    )

    # Locate the downloaded zip (Kaggle names it after the dataset slug)
    zip_name = DATASET_SLUG.split("/")[-1] + ".zip"
    zip_path = tmp_dir / zip_name

    if not zip_path.exists():
        # Fallback: find any zip in the tmp dir
        zips = list(tmp_dir.glob("*.zip"))
        if not zips:
            print("❌ No zip archive found after download!")
            sys.exit(1)
        zip_path = zips[0]

    print(f"📦 Extracting {zip_path.name} …")
    extract_dir = tmp_dir / "extracted"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"✅ Extraction complete → {extract_dir}")
    return extract_dir


# ─────────────────────────────────────────────────────────────
# Step 2 — Parse styles.csv
# ─────────────────────────────────────────────────────────────
def load_metadata(extract_dir: Path) -> pd.DataFrame:
    """
    Read ``styles.csv`` and return a clean DataFrame with
    columns ``['id', 'masterCategory']``.
    """
    # The CSV may be at the top level or inside a subdirectory
    csv_candidates = list(extract_dir.rglob("styles.csv"))
    if not csv_candidates:
        print("❌ Could not find styles.csv in the extracted archive!")
        sys.exit(1)

    csv_path = csv_candidates[0]
    print(f"\n📄 Reading metadata: {csv_path.name}")

    df = pd.read_csv(
        csv_path,
        on_bad_lines="skip",   # gracefully skip malformed rows
        usecols=["id", "masterCategory"],
    )

    # Drop rows with missing values
    initial = len(df)
    df.dropna(subset=["id", "masterCategory"], inplace=True)
    df["id"] = df["id"].astype(int)
    dropped = initial - len(df)

    print(f"   Total rows   : {initial:,}")
    if dropped:
        print(f"   Dropped (NaN): {dropped:,}")
    print(f"   Usable rows  : {len(df):,}")
    print(f"   Categories   : {df['masterCategory'].nunique()}")

    return df


# ─────────────────────────────────────────────────────────────
# Step 3 — Restructure into ImageFolder layout
# ─────────────────────────────────────────────────────────────
def restructure_images(df: pd.DataFrame, extract_dir: Path) -> None:
    """
    Move images from the flat extracted directory into::

        data/<masterCategory>/<id>.jpg
    """
    # Locate the images directory
    img_dirs = list(extract_dir.rglob("images"))
    img_dirs = [d for d in img_dirs if d.is_dir()]
    if not img_dirs:
        # Fallback: images might be at the root
        img_dirs = [extract_dir]

    images_root = img_dirs[0]
    print(f"\n🖼  Source images: {images_root}")

    # Clean existing data directory
    if DATA_DIR.exists():
        print(f"   Removing existing data/ directory …")
        shutil.rmtree(DATA_DIR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    moved   = 0
    missing = 0

    for _, row in df.iterrows():
        img_id   = int(row["id"])
        category = str(row["masterCategory"]).strip()
        src      = images_root / f"{img_id}.jpg"

        if not src.exists():
            missing += 1
            continue

        # Sanitize category name for filesystem
        safe_cat = category.replace("/", "-").replace("\\", "-")
        dest_dir = DATA_DIR / safe_cat
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / f"{img_id}.jpg"
        shutil.copy2(src, dest)
        moved += 1

    print(f"\n   ✅ Images moved : {moved:,}")
    print(f"   ⚠  Missing files: {missing:,}")


# ─────────────────────────────────────────────────────────────
# Step 4 — Prune minority classes
# ─────────────────────────────────────────────────────────────
def prune_minority_classes() -> None:
    """
    Remove category folders with fewer than ``MIN_IMAGES`` images.
    Prints a table of retained vs. removed categories.
    """
    print(f"\n🔍 Scanning categories (threshold: {MIN_IMAGES} images)")
    print(f"{'─' * 55}")

    retained  = []
    removed   = []

    for cat_dir in sorted(DATA_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue

        count = len(list(cat_dir.glob("*.jpg")))

        if count < MIN_IMAGES:
            removed.append((cat_dir.name, count))
            shutil.rmtree(cat_dir)
        else:
            retained.append((cat_dir.name, count))

    # ── Summary ─────────────────────────────────────────────
    if removed:
        print(f"\n  ❌ Removed ({len(removed)} classes, < {MIN_IMAGES} images):")
        for name, cnt in removed:
            print(f"     {name:<25s} → {cnt:>5,} images")

    print(f"\n  ✅ Retained ({len(retained)} classes):")
    print(f"  {'Category':<25s} {'Images':>8s}")
    print(f"  {'─' * 25}  {'─' * 8}")
    total = 0
    for name, cnt in sorted(retained, key=lambda x: -x[1]):
        print(f"  {name:<25s} {cnt:>8,}")
        total += cnt

    print(f"  {'─' * 25}  {'─' * 8}")
    print(f"  {'TOTAL':<25s} {total:>8,}")


# ─────────────────────────────────────────────────────────────
# Step 5 — Cleanup temporary files
# ─────────────────────────────────────────────────────────────
def cleanup(tmp_dir: Path) -> None:
    """Remove the temporary download/extraction directory."""
    print(f"\n🧹 Cleaning up: {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("   Done")


# ─────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """Run the full data ingestion pipeline."""
    print("=" * 55)
    print("  Smart E-commerce Product Classifier")
    print("  Data Ingestion & Preprocessing Pipeline")
    print("=" * 55)

    # Pre-flight check
    _check_kaggle_credentials()

    # Use a temporary directory for downloading
    tmp_dir = Path(tempfile.mkdtemp(prefix="ecom_data_"))

    try:
        # Step 1: Download & extract
        extract_dir = download_and_extract(tmp_dir)

        # Step 2: Parse metadata
        df = load_metadata(extract_dir)

        # Step 3: Restructure into ImageFolder layout
        restructure_images(df, extract_dir)

        # Step 4: Prune tiny classes
        prune_minority_classes()

    finally:
        # Step 5: Always clean up, even on errors
        cleanup(tmp_dir)

    print(f"\n{'=' * 55}")
    print(f"  ✅ Pipeline complete!")
    print(f"  Data ready at: {DATA_DIR}")
    print(f"  Next step:     python src/train.py")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
