#!/usr/bin/env sh
# This script converts the deepface data into lmdb format.

# set -e

DATA=CASIA_144/
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -r ./WebFace_train_$DBTYPE 
rm -r ./WebFace_val_$DBTYPE

../caffe-master/build/tools/convert_imageset --shuffle \
--resize_height=144 --resize_width=144 \
$DATA ./train.txt  ./WebFace_train_$DBTYPE

echo "Create deepID_train_$DBTYPE done."

../caffe-master/build/tools/convert_imageset --shuffle \
--resize_height=144 --resize_width=144 \
$DATA ./val.txt  ./WebFace_val_$DBTYPE

echo "Create WebFace_val_$DBTYPE done."
echo "Computing image mean..."

../caffe-master/build/tools/compute_image_mean -backend=$DBTYPE \
./WebFace_train_$DBTYPE ./mean.binaryproto

echo "Done."
