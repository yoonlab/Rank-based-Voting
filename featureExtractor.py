#!/usr/bin/env python

import _init_paths
import numpy as np
import os
import cv2
import pickle

from sklearn.externals import joblib


def get_proposals(extractor, detector, img, k=60, scale_factor=2.0, crop=None):
    im = cv2.imread(img)
    
    org_h, org_w = im.shape[:2]
    im = cv2.resize(im, (0,0), fx=1.0/scale_factor, fy=1.0/scale_factor)

    bbs, scores = detector.predict(im)

    topk_proposals = []
    im_area = im.shape[0]*im.shape[1]
    cands = []

    for i in range(len(bbs)):
        bb = bbs[i]
        obj = scores[i]
        area = (bb[2]-bb[0])*(bb[3]-bb[1])
        if area > im_area/32:
            cands.append( ( (obj+1)/(area**0.5), bb) )
    cands.sort(key=lambda x:x[0], reverse=True)
    cands = cands[:k]
    
    for i in range( len(cands) ):
        bb = cands[i][1]
        #extractor get windows with ymin, xmin, ymax, xmax
        topk_proposals.append( [bb[1]*scale_factor, bb[0]*scale_factor, min(bb[3]*scale_factor, org_h), min(bb[2]*scale_factor, org_w)] )
    
    if crop == None :
        # append the center region
        topk_proposals.append( [int(org_h/8),int(org_w/8),int(org_h*7/8),int(org_w*7/8)] )
    else :
        # append the confident region
        x1, y1, x2, y2 = crop[0], crop[1], crop[2], crop[3]
        topk_proposals.append( [y1, x1, y2, x2] )
        
    formatted_proposals = (img, np.array(topk_proposals))


    region_rel = []
    for i in range( len(topk_proposals) ):
        rel_dict = {}
        rel_dict['in'] = []
        rel_dict['out'] = []
        region_rel.append( rel_dict )

    for i in range( len(topk_proposals) ):
        yi1, xi1, yi2, xi2 = topk_proposals[i]
        for j in range(i+1, len(topk_proposals) ):
            yj1, xj1, yj2, xj2 = topk_proposals[j]
            if xi1-10 <= xj1 <= xj2 <= xi2+10 and yi1-10 <= yj1 <= yj2 <= yi2+10 :
                region_rel[i]['in'].append( j )
                region_rel[j]['out'].append( i )
            elif xj1-10 <= xi1 <= xi2 <= xj2+10 and yj1-10 <= yi1 <= yi2 <= yj2+10 :
                region_rel[i]['out'].append( j )
                region_rel[j]['in'].append( i )
    
    for i in range( len(topk_proposals) ):
        # give higher confidence for the regions related with confident region
        if crop != None and (i in region_rel[-1]['in'] or i == (len(topk_proposals)-1)) :
            region_rel[i]['confidence'] = 4.0
        else :
            region_rel[i]['confidence'] = 1.0

    return formatted_proposals, region_rel


def post_process(features, pca):
    # L2 normalize
    for i in range(len(features)):
        features[i] /= np.linalg.norm( features[i] )
    # PCA
    features = pca.transform( features )
    # L2 re-normalize
    for i in range(len(features)):
        features[i] /= np.linalg.norm( features[i] )
    return features


def reduce_rel(rel):
    for i in range(len(rel)):
        new_in = []
        for j in range(len(rel[i]['in'])):
            i_in = rel[i]['in'][j]
            for k in range(len(rel[i_in]['out'])):
                if rel[i_in]['out'][k] in rel[i]['in']:
                    break
                if k == len(rel[i_in]['out'])-1 :
                    new_in.append(i_in)
        rel[i]['in'] = new_in

        new_out = []
        for j in range(len(rel[i]['out'])):
            i_out = rel[i]['out'][j]
            for k in range(len(rel[i_out]['in'])):
                if rel[i_out]['in'][k] in rel[i]['out']:
                    break
                if k == len(rel[i_out]['in'])-1 :
                    new_out.append(i_out)
        rel[i]['out'] = new_out
    return rel



