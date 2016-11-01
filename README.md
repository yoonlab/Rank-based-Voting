# Rank-based-Voting

## Introduction
This project is the implementation of the paper which was submitted to Eurographics 2017.
It contains feature extraction using object detection and rank-based voting which are the main contributions of the paper.
Since this version of implementation does not utilize Product Quantization, we demonstrate the reduced experimentation with k=30 on Oxford 5k benchmark (k=150 in the paper).

## Requirements
Caffe deep learning framework (https://github.com/BVLC/caffe), python implementation of BING Objectness estimation (https://github.com/alessandroferrari/BING-Objectness), and OpenCV 2.4.xx are required for feature extraction. NumPy and ScikitLearn are also required. We used pre-train CaffeNet and VGG-19 networks and they can be found in https://github.com/BVLC/caffe/wiki/Model-Zoo.

## Usage

### Feature Extraction
1. Specify pathes for dataset, modules, and setting files in \_init_paths.py file. sample directory contains some examples for the format of file lists.
2. Execute the extract_test_img.sh for extracting features from test images.
3. Execute the extract_query_img.py for extracting features from query images.

### Search
1. Execute the search.py file.
2. Check the search results in the results directory.

## Notes
1. Existing results were achieved from Oxford 5k with k=30, and it shows 0.663 mAP (0.768 mAP with k=150 in the paper).
2. You can change the parameters k and v in extract_query_img.py and extract_test_img.py. But as we mentioned, this version does not adopt PQ so the larger k can increase the search time a lot.
3. As described in the paper, you can apply PQ for this method. You can generate binary codes after the feature extraction step, and achieve shortlists using binary codes previous to the search step.
4. learned_PCA.pkl in data directory is trained on Oxford 5k benchmark with single global descriptors, and whitening is not activated.
