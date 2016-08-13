#!/bin/bash
#
# Copyright (c) 2016, David Hirvonen
# All rights reserved.

# get script dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FACELABDIR=${DIR}/..
ASSETDIR=${FACELABDIR}/assets

DEST_REGRESSOR_URL="https://github.com/cheind/dest/releases/download/v0.8/dest_tracker_VJ_ibug.bin"
DEST_REGRESSOR=${DEST_REGRESSOR_URL##*/}

OPENCV_DETECTOR_URL="https://raw.githubusercontent.com/Itseez/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml"
OPENCV_DETECTOR=${OPENCV_DETECTOR_URL##*/}

if [ ! -f ${ASSETDIR}/${DEST_REGRESSOR} ]; then
    (cd $ASSETDIR && wget $DEST_REGRESSOR_URL)
fi

if [ ! -f ${ASSETDIR}/${OPENCV_DETECTOR} ]; then
    (cd $ASSETDIR && wget $OPENCV_DETECTOR_URL)
fi
