from pathlib import Path
import shutil
from geo_seg.train import train
from geo_seg.config import default_cfg


def test_train_smoke(tmp_path: Path):
    cfg = default_cfg
    cfg.train.epochs = 1
    cfg.runtime.out_dir = str(tmp_path / "run")
    best = train(cfg)
    assert Path(best).exists()
    # clean up created directory to keep workspace tidy (pytest tmp handles this)
    shutil.rmtree(cfg.runtime.out_dir, ignore_errors=True)
