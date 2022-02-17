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
        super(SimSiamKD_PoswMinT, self).__init__(
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

@ALGORITHMS.register_module()
class SimDis_Siam_nogt_sim_pred(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_Siam_nogt_sim_pred, self).__init__(
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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False)
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False)

        # simsiam_loss = 0.5 * (cosine_sim(p1, z2) + cosine_sim(p2, z1))
        simsiam_loss = 0.0
        distillation_loss1 = cosine_sim(p1, pt1)
        distillation_loss2 = cosine_sim(p2, pt2)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = simsiam_loss + distillation_loss

        return dict(loss=losses)


@ALGORITHMS.register_module()
class SimDis_Siam_nogt_sim_proj(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_Siam_nogt_sim_proj, self).__init__(
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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        # pt1 = self.teacher.head(zt1, zt2, loss_cal=False)
        # pt2 = self.teacher.head(zt2, zt1, loss_cal=False)

        # simsiam_loss = 0.5 * (cosine_sim(p1, z2) + cosine_sim(p2, z1))
        simsiam_loss = 0.0
        distillation_loss1 = cosine_sim(p1, zt1)
        distillation_loss2 = cosine_sim(p2, zt2)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = simsiam_loss + distillation_loss

        return dict(loss=losses)

@ALGORITHMS.register_module()
class SimDis_Siam_nogt_cross_pred(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_Siam_nogt_cross_pred, self).__init__(
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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False)
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False)

        # simsiam_loss = 0.5 * (cosine_sim(p1, z2) + cosine_sim(p2, z1))
        simsiam_loss = 0.0
        distillation_loss1 = cosine_sim(p1, pt2)
        distillation_loss2 = cosine_sim(p2, pt1)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = simsiam_loss + distillation_loss

        return dict(loss=losses)


@ALGORITHMS.register_module()
class SimDis_Siam_nogt_cross_proj(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_Siam_nogt_cross_proj, self).__init__(
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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)


        # simsiam_loss = 0.5 * (cosine_sim(p1, z2) + cosine_sim(p2, z1))
        simsiam_loss = 0.0
        distillation_loss1 = cosine_sim(p1, zt2)
        distillation_loss2 = cosine_sim(p2, zt1)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = simsiam_loss + distillation_loss

        return dict(loss=losses)

@ALGORITHMS.register_module()
class SimDis_Siam_poskd_sim_pred(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_Siam_poskd_sim_pred, self).__init__(
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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False)
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False)
        student_cs1 = cosine_sim(p1, z2, False)
        student_cs2 = cosine_sim(p2, z1, False)

        teacher_cs1 = cosine_sim(pt1, zt2, False).detach()
        teacher_cs2 = cosine_sim(pt2, zt1, False).detach()
        loss_pos_kd = 0.5 * (nn.functional.mse_loss(student_cs1, teacher_cs1) +
                             nn.functional.mse_loss(student_cs2, teacher_cs2))

        # simsiam_loss = 0.5 * (cosine_sim(p1, z2) + cosine_sim(p2, z1))

        distillation_loss1 = cosine_sim(p1, pt1)
        distillation_loss2 = cosine_sim(p2, pt2)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = loss_pos_kd + distillation_loss

        return dict(loss=losses)


@ALGORITHMS.register_module()
class SimDis_Siam_poskd_cross_pred(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_Siam_poskd_cross_pred, self).__init__(
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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False)
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False)

        student_cs1 = cosine_sim(p1, z2, False)
        student_cs2 = cosine_sim(p2, z1, False)

        teacher_cs1 = cosine_sim(pt1, zt2, False).detach()
        teacher_cs2 = cosine_sim(pt2, zt1, False).detach()

        loss_pos_kd = 0.5 * (nn.functional.mse_loss(student_cs1, teacher_cs1) +
                             nn.functional.mse_loss(student_cs2, teacher_cs2))

        distillation_loss1 = cosine_sim(p1, pt2)
        distillation_loss2 = cosine_sim(p2, pt1)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = loss_pos_kd + distillation_loss

        return dict(loss=losses)


@ALGORITHMS.register_module()
class SimDis_Siam_poskd_sim_proj(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_Siam_poskd_sim_proj, self).__init__(
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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False)
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False)

        student_cs1 = cosine_sim(p1, z2, False)
        student_cs2 = cosine_sim(p2, z1, False)

        teacher_cs1 = cosine_sim(pt1, zt2, False).detach()
        teacher_cs2 = cosine_sim(pt2, zt1, False).detach()

        loss_pos_kd = 0.5 * (nn.functional.mse_loss(student_cs1, teacher_cs1) +
                             nn.functional.mse_loss(student_cs2, teacher_cs2))
        distillation_loss1 = cosine_sim(p1, zt1)
        distillation_loss2 = cosine_sim(p2, zt2)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = loss_pos_kd + distillation_loss

        return dict(loss=losses)

@ALGORITHMS.register_module()
class SimDis_Siam_poskd_cross_proj(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimDis_Siam_poskd_cross_proj, self).__init__(
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

        zt1 = self.teacher.encoder(img_v1)[0]
        zt2 = self.teacher.encoder(img_v2)[0]

        p1 = self.head(z1, z2, loss_cal=False)
        p2 = self.head(z2, z1, loss_cal=False)

        pt1 = self.teacher.head(zt1, zt2, loss_cal=False)
        pt2 = self.teacher.head(zt2, zt1, loss_cal=False)

        student_cs1 = cosine_sim(p1, z2, False)
        student_cs2 = cosine_sim(p2, z1, False)

        teacher_cs1 = cosine_sim(pt1, zt2, False).detach()
        teacher_cs2 = cosine_sim(pt2, zt1, False).detach()

        loss_pos_kd = 0.5 * (nn.functional.mse_loss(student_cs1, teacher_cs1) +
                             nn.functional.mse_loss(student_cs2, teacher_cs2))


        distillation_loss1 = cosine_sim(p1, zt2)
        distillation_loss2 = cosine_sim(p2, zt1)

        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        losses = loss_pos_kd + distillation_loss

        return dict(loss=losses)


@ALGORITHMS.register_module()
class SimSiamKDMinIter(SimSiamKD):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKDMinIter, self).__init__(backbone, neck, head, init_cfg, **kwargs)

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

        shape_loss = teacher_loss1.shape
        min_loss1 = teacher_loss1.min().repeat(shape_loss)
        min_loss2 = teacher_loss2.min().repeat(shape_loss)


        losses = 0.5 * (nn.functional.mse_loss(self.head(z1, z2)['cossim'], min_loss1) +
                        nn.functional.mse_loss(self.head(z2, z1)['cossim'], min_loss2))
        return dict(loss=losses)



@ALGORITHMS.register_module()
class SimSiamKDMinEpoch(BaseModel):
    """SimSiam.

    Implementation of `Exploring Simple Siamese Representation Learning
    <https://arxiv.org/abs/2011.10566>`_.
    The operation of fixing learning rate of predictor is in
    `core/hooks/simsiam_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiamKDMinEpoch, self).__init__(init_cfg)
        assert neck is not None
        self.encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.backbone = self.encoder[0]
        self.neck = self.encoder[1]
        assert head is not None
        self.head = build_head(head)
        self.teacher = None
        # self.teacher = build_algorithm(teacher)
        # self.teacher.eval()
        # load_checkpoint(self.teacher, teacher_path, None, strict=False, revise_keys=[(r'^module.', '')])
        # print(self.teacher.state_dict())
        # exit(0)



    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

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
        img_v1 = img[0]
        img_v2 = img[1]

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        # zt1 = self.teacher.encoder(img_v1)[0]
        # zt2 = self.teacher.encoder(img_v2)[0]
        #
        # teacher_loss1 = self.teacher.head(zt1, zt2)['cossim'].detach()
        # teacher_loss2 = self.teacher.head(zt2, zt1)['cossim'].detach()
        s_sim1 = self.head(z1, z2)['cossim']
        s_sim2 = self.head(z2, z1)['cossim']
        teacher_sim = self.teacher.repeat(s_sim1.shape).float()
        losses = 0.5 * (nn.functional.mse_loss(s_sim1, teacher_sim) +
                        nn.functional.mse_loss(s_sim2, teacher_sim))
        return dict(loss=losses)

    def train_step(self, data, optimizer, teacher_minepoch):
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

        self.teacher = teacher_minepoch[0]
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if isinstance(data['img'], list):
            num_samples = len(data['img'][0].data)
        else:
            num_samples = len(data['img'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)
        return outputs