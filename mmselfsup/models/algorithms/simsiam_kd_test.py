# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import time
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck, build_algorithm
from mmcv.runner.checkpoint import load_checkpoint
from .base import BaseModel
import torch
import torch.nn.functional as F
from mmselfsup.models.algorithms.simsiam_kd import *


@ALGORITHMS.register_module()
class SimSiamKD_PoswMin(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKD_PoswMin, self).__init__(
            backbone,
            neck=neck,
            head=head,
            init_cfg=init_cfg,
            **kwargs
        )


    def forward_train(self, img):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
        Returns:
            loss[str, Tensor]: A dictionary of loss components
        """
        assert isinstance(img, list)
        self.teacher.eval()
        img_v1 = img[0]
        img_v2 = img[1]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        student_output1 = self.head(z1, z2)
        student_output2 = self.head(z2, z1)

        teacher_loss1 = torch.ones_like(student_output1['cossim']) * -1.0
        teacher_loss2 = torch.ones_like(student_output2['cossim']) * -1.0

        loss_kd_pos = 0.5 * (nn.functional.mse_loss(student_output1['cossim'], teacher_loss1) +
                        nn.functional.mse_loss(student_output2['cossim'], teacher_loss2))
        # loss_student = 0.5 * (student_output1['loss'] + student_output2['loss'])
        losses = loss_kd_pos
        return dict(loss=losses)

@ALGORITHMS.register_module()
class SimSiamKD_PoswMinT(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKD_PoswMin, self).__init__(
            backbone,
            neck=neck,
            head=head,
            init_cfg=init_cfg,
            **kwargs
        )


    def forward_train(self, img):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
        Returns:
            loss[str, Tensor]: A dictionary of loss components
        """
        assert isinstance(img, list)
        self.teacher.eval()
        img_v1 = img[0]
        img_v2 = img[1]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        student_output1 = self.head(z1, z2)
        student_output2 = self.head(z2, z1)

        teacher_loss1 = torch.ones_like(student_output1['cossim']) * -0.9458
        teacher_loss2 = torch.ones_like(student_output2['cossim']) * -0.9458

        loss_kd_pos = 0.5 * (nn.functional.mse_loss(student_output1['cossim'], teacher_loss1) +
                        nn.functional.mse_loss(student_output2['cossim'], teacher_loss2))
        # loss_student = 0.5 * (student_output1['loss'] + student_output2['loss'])
        losses = loss_kd_pos
        return dict(loss=losses)