# Copyright (c) 2015, David Hirvonen
# All rights reserved.

cmake_minimum_required(VERSION 3.0)

### Hunter snapshot that will be used ###
include("cmake/HunterGate.cmake")
HunterGate(
  URL "https://github.com/ruslo/hunter/archive/v0.14.12.tar.gz"
  SHA1 "2d444140db0d53393b5490604006bb4f4896a444"
  LOCAL
)

project(facelab VERSION 0.1.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(is_android ${ANDROID}) # syntax compatibility
string(COMPARE EQUAL "${CMAKE_OSX_SYSROOT}" "iphoneos" is_ios)
string(COMPARE EQUAL "${CMAKE_SYSTEM_NAME}" "Linux" is_linux)
string(COMPARE EQUAL "${CMAKE_GENERATOR}" "Xcode" is_xcode)

if(is_xcode)
  option(FACELAB_DISABLE_DSYM "Disable generation of dSYM information" ON)
  if(FACELAB_DISABLE_DSYM)
    set(CMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf")
  endif()
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release;MinSizeRel;RelWithDebInfo" CACHE STRING "Configuration types")
  set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)
endif()


## #################################################################
## Dependencies - OpenCV
## #################################################################
#hunter_add_package(Boost)
#find_package(Boost CONFIG REQUIRED)

hunter_add_package(Boost COMPONENTS system filesystem)
find_package(Boost CONFIG REQUIRED system filesystem)

## #################################################################
## Dependencies - OpenCV
## #################################################################

hunter_add_package(OpenCV)
find_package(OpenCV REQUIRED)
## #################################################################
## Dependencies - Eigen
## #################################################################

hunter_add_package(Eigen)
find_package(Eigen REQUIRED) 
message("EIGEN3_INCLUDE_DIR = ${EIGEN3_INCLUDE_DIR}")

## #################################################################
## Dependencies - cereal
## #################################################################

hunter_add_package(cereal)
find_package(cereal REQUIRED) 

## #################################################################
## Project
## #################################################################

include_directories(.)

add_subdirectory(src)
