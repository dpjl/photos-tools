#!/usr/bin/env python3
"""Migration script: convert old export format (4 directories) to new versioned exports.

Old format (per directory):
    {stem}.tiff              — exported image
    {stem}.result.json       — old sidecar with step log

New format (single output directory):
    {stem}.export.{NNN}.tiff          — versioned export image
    {stem}.export.{NNN}.recipe.json   — versioned recipe (v2)
    {stem}.export.{NNN}.mask.png      — mask from source (if inpaint enabled)

Each old directory becomes one export version (001=4a, 002=4b, 003=4c, 004=4d).
"""

import json
import os
import shutil
import sys

BASE = "/data/photos/1985 - all"
SOURCE_DIR = os.path.join(BASE, "[3] 1985 - Dated")
OUTPUT_DIR = os.path.join(BASE, "[4] 1985 - Renovated")

# Ordered: each directory becomes export index 1, 2, 3, 4
OLD_DIRS = [
    ("[4a] 1985 - Renovated 1", 1),
    ("[4b] 1985 - Renovated 2", 2),
    ("[4c] 1985 - Renovated 3", 3),
    ("[4d] 1985 - Renovated 4", 4),
]


def convert_result_json_to_recipe(result_data: dict, stem: str, index: int,
                                  has_mask: bool) -> dict:
    """Convert old result.json to new recipe v2 format."""
    steps = result_data.get("steps", [])

    step_order = [s["id"] for s in steps]
    step_enabled = {s["id"]: s["enabled"] for s in steps}
    step_params = {s["id"]: s.get("params", {}) for s in steps}

    # Extract wb_pick from the wb step if present
    wb_pick = None
    wb_patch_radius = 5
    for s in steps:
        if s["id"] == "wb":
            if "wb_pick" in s:
                wb_pick = s["wb_pick"]
            if "patch_radius" in s.get("params", {}):
                wb_patch_radius = s["params"]["patch_radius"]

    recipe = {
        "version": 2,
        "exported_at": result_data.get("processed_at", ""),
        "export_index": index,
        "source_filename": f"{stem}.tiff",
        "customized": True,
        "step_order": step_order,
        "step_enabled": step_enabled,
        "step_params": step_params,
        "wb_pick": wb_pick,
        "wb_patch_radius": wb_patch_radius,
        "has_mask": has_mask,
        "has_redeye_mask": False,
    }
    return recipe


def migrate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats = {"images": 0, "recipes": 0, "masks": 0, "skipped": 0, "errors": 0}

    for dir_name, export_index in OLD_DIRS:
        old_dir = os.path.join(BASE, dir_name)
        if not os.path.isdir(old_dir):
            print(f"  SKIP: {dir_name} not found")
            continue

        print(f"\n── {dir_name} → export index {export_index:03d} ──")

        # List all .tiff images
        images = sorted(f for f in os.listdir(old_dir) if f.lower().endswith(".tiff"))
        for img_name in images:
            stem = os.path.splitext(img_name)[0]
            ext = ".tiff"

            # Source files
            old_img = os.path.join(old_dir, img_name)
            old_json = os.path.join(old_dir, f"{stem}.result.json")

            # Target files
            new_img = os.path.join(OUTPUT_DIR, f"{stem}.export.{export_index:03d}{ext}")
            new_recipe = os.path.join(OUTPUT_DIR, f"{stem}.export.{export_index:03d}.recipe.json")
            new_mask = os.path.join(OUTPUT_DIR, f"{stem}.export.{export_index:03d}.mask.png")

            # Skip if already migrated
            if os.path.exists(new_img):
                stats["skipped"] += 1
                continue

            # Copy image
            shutil.copy2(old_img, new_img)
            stats["images"] += 1

            # Convert result.json → recipe.json
            if os.path.exists(old_json):
                with open(old_json, encoding="utf-8") as f:
                    result_data = json.load(f)

                # Check if inpaint is enabled
                inpaint_enabled = False
                for s in result_data.get("steps", []):
                    if s["id"] == "inpaint" and s.get("enabled", False):
                        inpaint_enabled = True
                        break

                # Copy mask from source directory if inpaint is enabled
                has_mask = False
                if inpaint_enabled:
                    source_mask = os.path.join(SOURCE_DIR, f"{stem}.mask.png")
                    if os.path.exists(source_mask):
                        shutil.copy2(source_mask, new_mask)
                        has_mask = True
                        stats["masks"] += 1

                recipe = convert_result_json_to_recipe(
                    result_data, stem, export_index, has_mask
                )
                with open(new_recipe, "w", encoding="utf-8") as f:
                    json.dump(recipe, f, ensure_ascii=False, indent=2)
                stats["recipes"] += 1
            else:
                print(f"  WARN: no result.json for {img_name}")
                stats["errors"] += 1

    print(f"\n{'='*60}")
    print(f"Migration terminée → {OUTPUT_DIR}")
    print(f"  Images copiées:  {stats['images']}")
    print(f"  Recipes créées:  {stats['recipes']}")
    print(f"  Masques copiés:  {stats['masks']}")
    print(f"  Déjà migrés:    {stats['skipped']}")
    print(f"  Erreurs:        {stats['errors']}")


if __name__ == "__main__":
    migrate()
