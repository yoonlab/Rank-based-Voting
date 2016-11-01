#!/bin/bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/OpenBLAS/lib:/usr/local/cuda/lib64"

for ((i=200;i<=5200;i+=200));do
	python extract_test_img.py --idx $i --gpu
	sync
done
