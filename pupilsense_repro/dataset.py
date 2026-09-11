from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import EyePreprocessor


def build_index(data_root: Path, eye: str = "left") -> pd.DataFrame:
    data_root = Path(data_root)
    pupil_col = f"{eye}_pupil"
    session_csvs = sorted(data_root.glob("*/*/session_data.csv"))
    frames = []
    for csv_path in session_csvs:
        session_dir = csv_path.parent
        participant_id = int(session_dir.parent.name)
        session_id = int(session_dir.name)
        table = pd.read_csv(csv_path)
        image_paths = [str(data_root / rel) for rel in table["frame_path"]]
        existing = [(rel, img, val)
                    for rel, img, val in zip(table["frame_path"], image_paths, table[pupil_col])
                    if Path(img).exists()]
        for rel, img, val in existing:
            frames.append({
                "participant_id": participant_id,
                "session_id": session_id,
                "frame_path": rel,
                "image_path": img,
                "true": float(val),
            })
    return pd.DataFrame(frames)


class EyeDentifyDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, preprocessor: EyePreprocessor):
        self._index = index_df.reset_index(drop=True)
        self._preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        row = self._index.iloc[i]
        tensor = self._preprocessor(Image.open(row["image_path"]))
        return tensor, float(row["true"]), int(row["participant_id"]), int(row["session_id"]), row["frame_path"]
