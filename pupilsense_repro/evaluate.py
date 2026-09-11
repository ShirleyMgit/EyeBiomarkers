from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .preprocessing import EyePreprocessor


def run_inference(model, dataset, batch_size: int = 128, device: str = "cpu") -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    records = []
    model.eval()
    with torch.no_grad():
        for tensors, trues, participant_ids, session_ids, frame_paths in tqdm(loader, desc="Predicting"):
            preds = model(tensors.to(device)).squeeze(1).cpu().tolist()
            batch = zip(participant_ids.tolist(), session_ids.tolist(), frame_paths,
                        trues.tolist(), preds)
            records.extend({
                "participant_id": pid, "session_id": sid, "frame_path": fp,
                "true": true, "prediction": pred,
            } for pid, sid, fp, true, pred in batch)
    return pd.DataFrame(records, columns=["participant_id", "session_id", "frame_path", "true", "prediction"])


def predict_diameter(pil_image: Image.Image, model, preprocessor: EyePreprocessor, device: str = "cpu") -> float:
    model.eval()
    with torch.no_grad():
        tensor = preprocessor(pil_image).unsqueeze(0).to(device)
        return float(model(tensor).item())
