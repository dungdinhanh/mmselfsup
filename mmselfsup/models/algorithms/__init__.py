# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseModel
from .byol import BYOL
from .classification import Classification
from .deepcluster import DeepCluster
from .densecl import DenseCL
from .moco import MoCo
from .npid import NPID
from .odc import ODC
from .relative_loc import RelativeLoc
from .rotation_pred import RotationPred
from .simclr import SimCLR
from .simsiam import SimSiam
from .swav import SwAV
from .simsiam_kd import SimSiamKD, SimSiamKDZT, SimSiamKD_PredMatching, SimSiamKD_GT, SimSiamKD_wNeg,\
    SimDis_Siam_simplified, SimSiamKD_PoswNeg, SimDis_PoswNeg, SimDis_Pos, SimDis_wNeg
from .simsiam_kd_test import *
__all__ = [
    'BaseModel', 'BYOL', 'Classification', 'DeepCluster', 'DenseCL', 'MoCo',
    'NPID', 'ODC', 'RelativeLoc', 'RotationPred', 'SimCLR', 'SimSiam', 'SwAV', 'SimSiamKD', 'SimSiamKDZT',
    'SimSiamKD_PredMatching', 'SimSiamKD_GT', 'SimSiamKD_wNeg', 'SimDis_Siam_simplified', 'SimSiamKD_PoswNeg',
    'SimDis_PoswNeg', 'SimDis_Pos', 'SimDis_wNeg', 'SimSiamKD_PoswMin', 'SimSiamKD_PoswMinT'
]
