import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

def process_checkpoint(checkpoint_path):
    file_npz = os.path.join(checkpoint_path, "features", "svd.npz.npy")
    svd_log = np.load(file_npz)
    return svd_log

def process_checkpoints_list(checkpoints_list, file_name):
    for checkpoint in checkpoints_list:
        # do smth
        svd_log = process_checkpoint(checkpoint)
        plt.plot(svd_log, label=os.path.basename(checkpoint))
        pass
    plt.legend()
    plt.savefig(os.path.join("visualization", file_name))
    plt.close()


if __name__ == '__main__':
    checkpoints_path='visualization/work_dirs/selfsup/simsiam/simsiam_resnet18_2xb128-coslr-200e_in30p'
    # checkpoints_list = list(glob(os.path.join(checkpoints_path, "*")))
    # print(checkpoints_list)
    checkpoints_list1 = [
        os.path.join(checkpoints_path, "simsiam"),
        os.path.join(checkpoints_path, "kd_imp"),
        os.path.join(checkpoints_path, "simdis")
    ]
    process_checkpoints_list(checkpoints_list1, "implicit.png")

    checkpoints_list2 = [
        os.path.join(checkpoints_path, "simsiam"),
        os.path.join(checkpoints_path, "simdis")
    ]
    process_checkpoints_list(checkpoints_list2, "simdis.png")

    checkpoints_list3 = [
        os.path.join(checkpoints_path, "simsiam"),
        os.path.join(checkpoints_path, "kd_exp"),
        os.path.join(checkpoints_path, "simdis")
    ]
    process_checkpoints_list(checkpoints_list3, "explicit.png")

    checkpoints_list4 = [
        os.path.join(checkpoints_path, "simsiam"),
        os.path.join(checkpoints_path, "kd_imp"),
        os.path.join(checkpoints_path, "kd_exp"),
        os.path.join(checkpoints_path, "kd_exp_imp")
    ]
    process_checkpoints_list(checkpoints_list4, "impvsexp.png")

    checkpoints_list5 = [
        os.path.join(checkpoints_path, "simsiam"),
        os.path.join(checkpoints_path, "kd_exp_imp"),
        os.path.join(checkpoints_path, "simdis")
    ]
    process_checkpoints_list(checkpoints_list5, "simdisvsimpexp.png")

    pass