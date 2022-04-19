_base_ = [
    '../_test_/models/resnet18_simsiam_kd_olmh_dimcollapse.py',
    '../_test_/datasets/imagenet10p_mocov2_wori_b64_cluster.py',
    '../_test_/schedules/sgd_coslr-200e_in30p_kd.py',
    '../_test_/default_runtime.py',
]

# set base learning rate
lr = 0.05

# additional hooks
custom_hooks = [
    dict(type='SimSiamHook', priority='HIGH', fix_pred_lr=True, lr=lr)
]

# optimizer
optimizer = dict(lr=lr, paramwise_options={'predictor': dict(fix_lr=True)})

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)


# runtime settings
runner = dict(type='KDBasedRunnerSaveImagesAll', max_epochs=1)

load_from = "../../scratch/dso/openss/work_dirs/selfsup/simsiam_kd_test/simsiamolmh/epoch_150.pth"




