import logging
import os
from contextlib import contextmanager
from pathlib import Path

from omegaconf import Container

logger = logging.getLogger(__name__)


@contextmanager
def nlp_cluster(cfg: Container):
    """Set the cluster for NLP stanford."""

    user_path = Path(f"/scr/biggest/{cfg.user}")
    user_path.mkdir(exist_ok=True)

    # project path on current machine
    proj_path = user_path / cfg.project_name
    proj_path.mkdir(exist_ok=True)
    cfg.paths.tmp_dir = str(proj_path)
    logger.info(f"TMP dir: {cfg.paths.tmp_dir}.")

    prev_work_dir = os.getcwd()
    curr_work_dir = user_path / str(cfg.paths.relative_work)
    curr_work_dir.mkdir(exist_ok=True, parents=True)
    os.chdir(curr_work_dir)  # change directory on current machine

    try:
        yield
    finally:
        os.chdir(prev_work_dir)  # put back old dir to be safe


