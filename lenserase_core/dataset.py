"""
Dataset and Loader Utilities for LensErase
Reference: Section 2.1 (LensErase-100M) and Section 2.3 (LensErase-Bench)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

class LensEraseDataset:
    """
    Dataset loader for LensErase official benchmark cases and showcase data.
    Ensures that masks and all baseline images are consistently resized to match
    the source input image geometry.
    """
    def __init__(self, root_dir: Optional[str] = None, group: Optional[str] = None):
        if root_dir is None:
            self.root_dir = Path(__file__).resolve().parent.parent
        else:
            self.root_dir = Path(root_dir)

        unified_path = self.root_dir / "data" / "unified_dataset" / "dataset_index.json"
        official_path = self.root_dir / "data" / "lenserase_official" / "dataset_index.json"

        if unified_path.exists():
            self.index_path = unified_path
        elif official_path.exists():
            self.index_path = official_path
        else:
            raise FileNotFoundError(
                f"Dataset index not found at {unified_path} or {official_path}."
            )

        with open(self.index_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        records = self.metadata.get("records", [])
        if group:
            records = [r for r in records if r.get("group") == group]
        self.records: List[Dict[str, Any]] = records

    def __len__(self) -> int:
        return len(self.records)

    def get_categories(self) -> List[str]:
        cats = sorted(list({r.get("category") for r in self.records if r.get("category")}))
        return cats

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        files = rec.get("files", {})

        # Load input image
        input_path = self.root_dir / files["Input"]
        input_pil = Image.open(input_path).convert("RGB")
        target_w, target_h = input_pil.size
        input_np = np.array(input_pil)

        # Load mask and guarantee exact spatial alignment with input image
        mask_path = self.root_dir / files["Mask"]
        mask_pil = Image.open(mask_path).convert("L")
        if mask_pil.size != (target_w, target_h):
            mask_pil = mask_pil.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        mask_np = (np.array(mask_pil) > 128).astype(np.uint8)

        # Load LensErase output and guarantee exact spatial alignment
        lenserase_np = None
        if "LensErase" in files:
            le_path = self.root_dir / files["LensErase"]
            if le_path.exists():
                lenserase_pil = Image.open(le_path).convert("RGB")
                if lenserase_pil.size != (target_w, target_h):
                    lenserase_pil = lenserase_pil.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
                lenserase_np = np.array(lenserase_pil)

        # Load baseline outputs and guarantee exact spatial alignment
        baselines = {}
        for view_name, rel_path in files.items():
            if view_name not in ("Input", "Mask", "LensErase"):
                b_path = self.root_dir / rel_path
                if b_path.exists():
                    b_pil = Image.open(b_path).convert("RGB")
                    if b_pil.size != (target_w, target_h):
                        b_pil = b_pil.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
                    baselines[view_name] = np.array(b_pil)

        return {
            "id": rec["id"],
            "title": rec["title"],
            "category": rec["category"],
            "group": rec["group"],
            "description": rec["description"],
            "headline": rec.get("headline", ""),
            "input": input_np,
            "mask": mask_np,
            "lenserase": lenserase_np,
            "baselines": baselines,
            "size": (target_h, target_w)
        }

def create_mask_overlay(image: np.ndarray, mask: np.ndarray,
                        color: tuple = (255, 60, 60), alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a translucent colored highlight on masked regions.
    Safeguards against any dimensional mismatch.
    """
    if image.shape[:2] != mask.shape[:2]:
        m_pil = Image.fromarray(mask).resize((image.shape[1], image.shape[0]), resample=Image.Resampling.NEAREST)
        mask = np.array(m_pil)

    overlay = image.copy().astype(np.float32)
    m = (mask > 0.5)
    overlay[m] = overlay[m] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)
