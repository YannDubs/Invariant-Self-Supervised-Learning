import logging
import os
import platform
from contextlib import contextmanager
from pathlib import Path

from omegaconf import Container

from utils.helpers import remove_rf

logger = logging.getLogger(__name__)

def get_nlp_path(cfg: Container) -> Path:
    """Return (create if needed) path on current machine for NLP stanford."""
    machine_name = platform.node().split(".")[0]
    machine_path = Path(f"/{machine_name}/")

    user_paths = list(machine_path.glob(f"*/{cfg.user}"))
    if len(user_paths) == 0:
        possible_paths = [p for p in machine_path.iterdir() if "scr" in str(p)]
        user_path = possible_paths[-1] / str(cfg.user)
        user_path.mkdir()
    else:
        user_path = user_paths[-1]

    return user_path


@contextmanager
def nlp_cluster(cfg: Container):
    """Set the cluster for NLP stanford."""

    user_path = get_nlp_path(cfg)

    # project path on current machine
    proj_path = user_path / cfg.project_name
    proj_path.mkdir(exist_ok=True)
    cfg.paths.tmp_dir = str(proj_path)
    logger.info(f"TMP dir: {cfg.paths.tmp_dir}.")
    new_data_path = proj_path / "data"

    if not new_data_path.is_symlink():
        # make sure it's a symlink
        remove_rf(new_data_path, not_exist_ok=True)

        if (proj_path.parents[2] / "scr0").exists():
            # if possible symlink to scr0 as that's where most data is saved
            new_data_path.symlink_to(proj_path.parents[2] / "scr0")
        else:
            # if not just use current scr
            new_data_path.symlink_to(proj_path.parents[1])

    prev_work_dir = os.getcwd()
    curr_work_dir = user_path / str(cfg.paths.relative_work)
    curr_work_dir.mkdir(exist_ok=True, parents=True)
    os.chdir(curr_work_dir)  # change directory on current machine

    try:
        yield
    finally:
        os.chdir(prev_work_dir)  # put back old dir to be safe


