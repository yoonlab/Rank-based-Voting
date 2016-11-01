# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

caffe_dir = "/home/jaehyeong/caffe"

# Add caffe to PYTHONPATH
caffe_path = osp.join(caffe_dir, 'python')
add_path(caffe_path)

# Paths for BING
bing_dir = "/home/jaehyeong/Desktop/BING-Objectness-master/source"
bing_param_file="/home/jaehyeong/Desktop/BING-Objectness-master/doc/bing_params.json"
add_path(bing_dir)


# Paths for image data
data_dir = "/home/jaehyeong/Desktop/Datasets/oxford5k/"
img_list_file = "/home/jaehyeong/Desktop/Datasets/oxford5k/filenames.txt"

gt_dir = "/home/jaehyeong/Desktop/Datasets/oxford5k_gt"
query_list_file = "/home/jaehyeong/Desktop/Datasets/oxford5k_query_list.txt"


