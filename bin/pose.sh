#!/bin/bash
#
# Copyright (c) 2016, David Hirvonen
# All rights reserved.

PATH="${PWD}:${PWD}/_install/libcxx/bin:${PATH}"

FACELABDIR=${PWD}
ASSETDIR=${FACELABDIR}/assets

ARGUMENTS=(
    "--input=${HOME}/Downloads/face.jpg "
    "--output=/tmp/pose.jpg "
    "--regressor=${ASSETDIR}/dest_tracker_VJ_ibug.bin "
    "--detector=${ASSETDIR}/haarcascade_frontalface_alt.xml "
    "--model=${FACELABDIR}/src/3rdparty/eos/share/sfm_shape_3448.bin "
    "--mapping=${FACELABDIR}/src/3rdparty/eos/share/ibug2did.txt "
)

eval pose "${ARGUMENTS[*]}"
