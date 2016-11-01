
import os
import numpy as np
import cv2
import time
import pickle

from _init_paths import *
from multiprocessing import Pool

        
def voting(query, query_rel, testset, indexes, index_dict, rel_dict, n=10):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=10)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(np.asarray(query,np.float32),np.asarray(testset,np.float32),k=10*n)

    rel_thres = []
    for i in range(len(query)):
        rel_thres.append( matches[i][n].distance )
    
    votes_dict = {}
    for i in range(len(query)):
        votes = {}
        j = 0
        while len(votes.keys()) < n :
            matched_idx = matches[i][j].trainIdx
            img_num = indexes[ matched_idx ][0]
            feature_idx = indexes[ matched_idx ][1]

            rel_s = query_rel[i]['confidence']
            for x in query_rel[i]['in']:
                for y in rel_dict[img_num][feature_idx]['in']:
                    idx = index_dict[img_num] + y
                    rel_feature = testset[idx]
                    if np.linalg.norm(query[x]-rel_feature) < rel_thres[x] :
                        rel_s += query_rel[i]['confidence']
                        break

            for x in query_rel[i]['out']:
                for y in rel_dict[img_num][feature_idx]['out']:
                    idx = index_dict[img_num] + y
                    rel_feature = testset[idx]
                    if np.linalg.norm(query[x]-rel_feature) < rel_thres[x] :
                        rel_s += query_rel[i]['confidence']
                        break
            
            
            if img_num in votes.keys():
                # rel_s * rank_s
                score = rel_s / len(votes.keys())
                if votes[img_num] < score :
                    votes[img_num] = score
            else :
                # rel_s * rank_s
                score = rel_s / (len(votes.keys())+1)
                votes[img_num] = score

            j += 1
            if j == len(matches[i]):
                break
        
        
        for v in votes.keys():
            if votes_dict.has_key( v ):
                votes_dict[v] += votes[v]
            else :
                votes_dict[v] = votes[v]
                
    voting_res = []
    for k in votes_dict.keys():
        voting_res.append( (k, votes_dict[k]) )
    voting_res.sort(key=lambda x:x[1], reverse=True )

    return voting_res
    

def load_features():
    features = []
    for i in range(200,5200+1,200):
        f = open("data/features/%d.pkl"%i,"rb")
        features_partial = pickle.load(f)
        f.close()
        for feature in features_partial :
            features.append(feature)
        
    return features
	
def load_indexes():
    indexes = []
    index_dict = {}
    for i in range(200,5200+1,200):
        f = open("data/indexes/%d.pkl"%i,"rb")
        indexes_partial = pickle.load(f)
        f.close()
        for index in indexes_partial :
            if index[1] == 0 :
                index_dict[index[0]] = len(indexes)
            indexes.append(index)
                
    return indexes, index_dict
	
def load_relations():
    relations = {}
    for idx in range(200,5200+1,200):
        f = open("data/relations/%d.pkl"%idx,"rb")
        relations_partial = pickle.load(f)
        f.close()
        relations.update(relations_partial)

    return relations
	
def voting_mult(args):
    query, query_rel, features, indexes, index_dict, rel_dict, votes_per_region = args
    return voting(query, query_rel, features, indexes, index_dict, rel_dict, n=votes_per_region)

def search():
    print "load data"
    start = time.time()
    features = load_features()
    print "  - loaded features, took %.3f sec"%(time.time()-start)
    start = time.time()
    indexes, index_dict = load_indexes()
    print "  - loaded indexes, took %.3f sec"%(time.time()-start)
    start = time.time()
    relations = load_relations()
    print "  - loaded relations, took %.3f sec"%(time.time()-start)

    print "load query"
    start = time.time()
    f = open("data/query.pkl","rb")
    queries = pickle.load(f)
    f.close()
    print "  - loaded queries, took %.3f sec"%(time.time()-start)

    vote_per_region = 250
    q_list = sorted(queries)
    args = []
    for q in q_list :
        query = queries[q]
        args.append( (query['feature'], query['relation'], features, indexes, index_dict, relations, vote_per_region) )
        
    print "start searching"
    start = time.time()
    pool = Pool()
    search_results = pool.map(voting_mult, args)
    pool.close()
    pool.join()
    print "  - finished searching, took %.3f sec for %d queries"%(time.time()-start,len(q_list))
    
    print "write result files"
    start = time.time()
    for i in range(len(q_list)):
        dist_lst = search_results[i]
        f = open("results/%s.txt" % q_list[i] ,"w")
        for j in range( len(dist_lst) ):
            f.write("%s\n" % dist_lst[j][0][:-4] )
        f.close()
    print "  - finished writing, took %.3f sec"%(time.time()-start)


if __name__ == "__main__":
	search()

    
    

