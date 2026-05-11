"""recommender_pkg: Two-Tower MLP distributed training package.

03-custom-package-script-based의 노트북들은 이 패키지를 wheel install 후 import한다.

    %pip install ./custom_packages/dist/recommender_pkg-0.1.0-py3-none-any.whl
    from recommender_pkg import TwoTowerMLP, train_fn, fit_lightning
"""

from .lightning_trainer import fit as fit_lightning
from .model import EarlyStopping, TwoTowerMLP
from .torch_distributor_trainer import train_fn

__version__ = "0.1.0"
__all__ = ["TwoTowerMLP", "EarlyStopping", "train_fn", "fit_lightning"]
