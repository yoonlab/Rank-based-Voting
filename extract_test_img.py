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

def extract_features(extractor, img_idx):
    bing_params = bing_param_setting(bing_param_file)
    bing_detector = Bing(bing_params['w_1st'],bing_params['sizes'],bing_params['w_2nd'],num_bbs_per_size_1st_stage=bing_params["num_win_psz"],num_bbs_final=bing_params["num_bbs"])

    pca = joblib.load("data/learned_PCA.pkl")

    relations = {}
    formatted_proposals = []
    indexes = []
    
    list_f = open(img_list_file)
    img_lst = list_f.read().split()
    img_lst = img_lst[img_idx-200:img_idx]
    
    for img_name in img_lst:
        img_name = img_name.strip()
        if img_name == "" or img_name[-3:] != "jpg":
            continue
        img = os.path.join(data_dir, img_name)

        # k : number of regions
        proposals, rels = get_proposals(extractor, bing_detector, img, k=30)

        for idx in range(len(proposals[1])):
            indexes.append( (img_name, idx) )
        formatted_proposals.append( proposals )
        relations[img_name] = reduce_rel(rels)

    features = extractor.extract_features(formatted_proposals, layer='fc6')
    features = post_process( features, pca )
    
    
    f = open("data/features/%d.pkl"%img_idx,"wb")
    pickle.dump( features, f )
    f.close()
    
    f = open("data/indexes/%d.pkl"%img_idx,"wb")
    pickle.dump( indexes, f )
    f.close()
    
    f = open("data/relations/%d.pkl"%img_idx,"wb")
    pickle.dump( relations, f )
    f.close()
    


if __name__ == "__main__":
    from network_settings import main_caffe, main_vgg
    main = main_vgg
    import sys
    extractor, idx = main(sys.argv)

    if idx >= 200 :
        extract_features(extractor, idx)
    else :
        print "idx >= 200"
    
