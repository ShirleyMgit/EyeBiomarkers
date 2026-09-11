from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def synthetic_data_root(tmp_path) -> Path:
    """Two participants, two sessions each, two frames each, with session_data.csv."""
    root = tmp_path / "left_eyes_data"
    rows_per = []
    for pid in (1, 2):
        for sid in (1, 2):
            session_dir = root / str(pid) / str(sid)
            session_dir.mkdir(parents=True)
            csv_rows = []
            for fi in (1, 2):
                fname = f"frame_{fi:02d}.png"
                Image.new("RGB", (32, 16), color=(100 + fi, 110, 120)).save(session_dir / fname)
                csv_rows.append({
                    "session_id": sid,
                    "left_pupil": 2.0 + 0.1 * fi + 0.5 * pid,
                    "right_pupil": 2.5 + 0.1 * fi,
                    "frame_path": f"{pid}/{sid}/{fname}",
                })
            pd.DataFrame(csv_rows).to_csv(session_dir / "session_data.csv", index=False)
    return root
