#!/bin/bash

######################################
########## get script dir ############
######################################

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. ${DIR}/download-assets.sh

TOOLCHAIN=ios-9-2
APP=_builds/${TOOLCHAIN}/src/app/filter/Debug-iphoneos/filter.app
BUNDLE_ID=com.elucideye.filter.debug

FACE_IMG=${HOME}/Downloads/23381735-8122-4116-938F-37166D85821B-5538-00000990A9246FC0.jpg

convert ${FACE_IMG} ${FACE_IMG%.*}.png
FACE_IMG=${FACE_IMG%.*}.png

DO_SETUP=1
if [ ${DO_SETUP} -gt 0 ]; then
    ios-deploy --list_bundle_id
    ios-deploy --justlaunch --bundle ${APP}

    ASSETS=(
        ${FACE_IMG}
        ${ASSETDIR}/${DEST_REGRESSOR}
        ${ASSETDIR}/${OPENCV_DETECTOR}
    )
    
    for TARGET in ${ASSETS[*]}; do
        echo "#################################"
        ios-deploy --bundle_id ${BUNDLE_ID} --upload ${TARGET} --to Documents/${TARGET##*/}
    done
    
    ios-deploy --bundle_id ${BUNDLE_ID} --mkdir "Documents/output/"
fi

ARGUMENTS=(\
    "--input=HOME/Documents/${FACE_IMG##*/} "
    "--output=HOME/Documents/output/cartoon.png "
    "--width=512 "
    "--regressor=HOME/Documents/dest_tracker_VJ_ibug.bin "
    "--detector=HOME/Documents/haarcascade_frontalface_alt.xml "
)

ios-deploy --bundle ${APP} --justlaunch --noinstall --args "${ARGUMENTS[*]}"
ios-deploy --bundle_id ${BUNDLE_ID} --list --verbose
ios-deploy --bundle_id ${BUNDLE_ID} --download=/Documents/output --to .
