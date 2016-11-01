import _init_paths
import argparse
import os
import caffe
import numpy as np
from extractor import Extractor

def main_caffe(argv):
    pycaffe_dir = _init_paths.caffe_path

    parser = argparse.ArgumentParser()
    # Required argument.
    parser.add_argument(
        "--idx",
        default="0"
    )

    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )
    parser.add_argument(
        "--context_pad",
        type=int,
        default='16',
        help="Amount of surrounding context to collect in input window."
    )
    args = parser.parse_args()

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
        if mean.shape[1:] != (1, 1):
            mean = mean.mean(1).mean(1)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    idx = int(args.idx)
    
    # Make detector.
    extractor = Extractor(args.model_def, args.pretrained_model, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap,
            context_pad=args.context_pad)
    return extractor, idx
    
def main_vgg(argv):
    pycaffe_dir = _init_paths.caffe_path

    parser = argparse.ArgumentParser()
    # Required argument.
    parser.add_argument(
        "--idx",
        default="0"
    )

    # Optional arguments.
    # ../models/bvlc_reference_caffenet/deploy.prototxt
    # ../models/placesCNN/places205CNN_deploy.prototxt
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/ILSVRC_19_layers/VGG_ILSVRC_19_layers_deploy.prototxt"),
        help="Model definition file."
    )
    
    # ../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
    # ../models/placesCNN/places205CNN_iter_300000.caffemodel
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    # imagenet/ilsvrc_2012_mean.npy
    # caffe/imagenet/places205CNN_mean.binaryproto
    parser.add_argument(
        "--mean_file",
        default=os.path.join(''),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )
    parser.add_argument(
        "--context_pad",
        type=int,
        default='16',
        help="Amount of surrounding context to collect in input window."
    )
    args = parser.parse_args()

    mean, channel_swap = None, None
    
    mean = np.array([103.939, 116.779, 123.68])
    
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    idx = int(args.idx)

    # Make detector.
    extractor = Extractor(args.model_def, args.pretrained_model, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap,
            context_pad=args.context_pad)
    return extractor, idx
    
    

