#!/usr/bin/env python

import _init_paths
import numpy as np
import os
import time
import cv2
import pickle

from bing import Bing, bing_param_setting
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from featureExtractor import *
from _init_paths import *

def ox5k_get_query(data_dir, query_name):
    query_file = os.path.join(data_dir, query_name+"_query.txt")
    f = open(query_file, "r")
    l = f.readline().split(' ')
    query = l[0]
    f.close()
    return query[5:] + ".jpg", [float(l[1]),float(l[2]),float(l[3]),float(l[4])]


def extract_features(extractor):
    bing_params = bing_param_setting(bing_param_file)
    bing_detector = Bing(bing_params['w_1st'],bing_params['sizes'],bing_params['w_2nd'],num_bbs_per_size_1st_stage=bing_params["num_win_psz"],num_bbs_final=bing_params["num_bbs"])

    pca = joblib.load("data/learned_PCA.pkl")

    query_list = open(query_list_file,"r")

    query_dict = {}

    for q in query_list:
        query_name = q.strip()
        if q == "":
            continue
        print query_name
            
        query, crop = ox5k_get_query(gt_dir, query_name)
        img = os.path.join(data_dir, query)
        
        proposals, rels = get_proposals(extractor, bing_detector, img, k=30, crop=crop)

        formatted_proposals = [proposals]
        features = extractor.extract_features(formatted_proposals, layer='fc6')
        features = post_process( features, pca )

        query_dict[query_name] = {}
        query_dict[query_name]["feature"] = features
        query_dict[query_name]["relation"] = reduce_rel(rels)

    query_list.close()


    f = open("data/query.pkl","wb")
    pickle.dump( query_dict, f )
    f.close()


if __name__ == "__main__":
    from network_settings import main_caffe, main_vgg
    main = main_vgg
    import sys
    extractor, idx = main(sys.argv)
    extract_features(extractor)
    

