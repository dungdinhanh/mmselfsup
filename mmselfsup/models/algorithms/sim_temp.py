# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import time
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck, build_algorithm
from mmcv.runner.checkpoint import load_checkpoint
from .base import BaseModel
import torch
import torch.nn.functional as F
from mmselfsup.models.algorithms.simsiam_kd import *
from collections import OrderedDict
import torch.distributed as dist




@ALGORITHMS.register_module()
class SimSiamKD_ignorelower(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKD_ignorelower, self).__init__(backbone, neck, head, init_cfg, **kwargs)


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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        teacher_loss1 = self.teacher.head(zt1, zt2)['cossim'].detach()
        teacher_loss2 = self.teacher.head(zt2, zt1)['cossim'].detach()

        student_loss1 = self.head(z1, z2)['cossim']
        student_loss2 = self.head(z2, z1)['cossim']

        # for plotting loss
        l_s1 = torch.mean(student_loss1)
        l_s2 = torch.mean(student_loss2)
        l_s = 1 / 2 * (l_s1 + l_s2).detach()
        l_t = 1 / 2 * (torch.mean(teacher_loss1) + torch.mean(teacher_loss2)).detach()

        index_lower1 = student_loss1 < teacher_loss1
        index_lower2 = student_loss2 < teacher_loss2

        student_loss1[index_lower1] *= 0
        student_loss2[index_lower2] *= 0

        losses = 0.5 * (nn.functional.mse_loss(student_loss1, teacher_loss1) +
                        nn.functional.mse_loss(student_loss2, teacher_loss2))
        return dict(loss=losses, l_student=l_s, l_teacher=l_t)


    def train_step(self, data, optimizer, teacher_model):
        if self.teacher is None:
            self.teacher = teacher_model
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if isinstance(data['img'], list):
            num_samples = len(data['img'][0].data)
        else:
            num_samples = len(data['img'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs


@ALGORITHMS.register_module()
class SimSiamKD_ILMH(SimSiamKD):  # ignore lower minimize Higher
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKD_ILMH, self).__init__(backbone, neck, head, init_cfg, **kwargs)

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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        teacher_loss1 = self.teacher.head(zt1, zt2)['cossim'].detach()
        teacher_loss2 = self.teacher.head(zt2, zt1)['cossim'].detach()

        student_loss1 = self.head(z1, z2)['cossim']
        student_loss2 = self.head(z2, z1)['cossim']

        # for plotting loss
        l_s1 = torch.mean(student_loss1)
        l_s2 = torch.mean(student_loss2)
        l_s = 1 / 2 * (l_s1 + l_s2).detach()
        l_t = 1 / 2 * (torch.mean(teacher_loss1) + torch.mean(teacher_loss2)).detach()

        index_lower1 = student_loss1 < teacher_loss1
        index_lower2 = student_loss2 < teacher_loss2

        student_loss1[index_lower1] *= 0
        student_loss2[index_lower2] *= 0

        losses = 0.5 * (torch.mean(student_loss1) + torch.mean(student_loss2))
        return dict(loss=losses, l_student=l_s, l_teacher=l_t)

    def train_step(self, data, optimizer, teacher_model):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        if self.teacher is None:
            self.teacher = teacher_model
        self.teacher = torch.min(teacher_minepoch[epoch[0]: epoch[0] + 5])
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if isinstance(data['img'], list):
            num_samples = len(data['img'][0].data)
        else:
            num_samples = len(data['img'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)
        return outputs


@ALGORITHMS.register_module()
class SimSiamKD_OLMH(SimSiamKD): # optimize lower, minimize Higher
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKD_OLMH, self).__init__(backbone, neck, head, init_cfg, **kwargs)

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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        teacher_loss1 = self.teacher.head(zt1, zt2)['cossim'].detach()
        teacher_loss2 = self.teacher.head(zt2, zt1)['cossim'].detach()

        student_loss1 = self.head(z1, z2)['cossim']
        student_loss2 = self.head(z2, z1)['cossim']

        # for plotting loss
        l_s1 = torch.mean(student_loss1)
        l_s2 = torch.mean(student_loss2)
        l_s = 1/2 * (l_s1 + l_s2).detach()
        l_t = 1/2 * (torch.mean(teacher_loss1) + torch.mean(teacher_loss2)).detach()

        index_lower1 = student_loss1 < teacher_loss1
        index_lower2 = student_loss2 < teacher_loss2

        index_higher1 = torch.bitwise_not(index_lower1)
        index_higher2 = torch.bitwise_not(index_lower2)

        loss1 = 0.5 * (torch.mean(student_loss1[index_higher1]) + torch.mean(student_loss2[index_higher2]))
        loss2 = 0.5 * (nn.functional.mse_loss(student_loss1[index_lower1], teacher_loss1[index_lower1]) +
                       nn.functional.mse_loss(student_loss2[index_lower2], teacher_loss2[index_lower2]))

        losses = loss1 + loss2
        return dict(loss=losses, l_student=l_s, l_teacher=l_t)
