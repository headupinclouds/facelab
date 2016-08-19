#!/bin/bash
#
# Copyright (c) 2016, David Hirvonen
# All rights reserved.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. ${DIR}/download-assets.sh

TOOLCHAIN=xcode
PATH="${DIR}/../_install/${TOOLCHAIN}/bin:${PATH}"

FACE_IMAGE=${HOME}/Downloads/face.jpg
if [ $# -ge 1 ] && [ -f ${1} ]; then
    FACE_IMAGE=${1}
fi
    
if [ ! -f ${FACE_IMAGE} ]; then
    2>&1 echo "Must specify valid face image"
    exit 1
fi

ARGUMENTS=(\
    "--input=${FACE_IMAGE} "
    "--output=/tmp/cartoon.jpg "
    "--width=512 "
    "--regressor=${ASSETDIR}/dest_tracker_VJ_ibug.bin "
    "--detector=${ASSETDIR}/haarcascade_frontalface_alt.xml "
    "--log=log.txt "
)

eval filter "${ARGUMENTS[*]}"
