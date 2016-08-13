# Copyright (c) 2015, David Hirvonen
# All rights reserved.

cmake_minimum_required(VERSION 3.0)

### Hunter snapshot that will be used ###
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.15.9.tar.gz"
    SHA1 "3e1da8447bfdbf9514f42abf8ff05455530b9b1c"
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

option(FACELAB_USE_EOS "Use EOS" NO)
if(FACELAB_USE_EOS)
  add_definitions(-DFACELAB_USE_EOS=1)

  # EOS dependencies
  include_directories("${CMAKE_SOURCE_DIR}/src/3rdparty/eos/include")
  include_directories("${CMAKE_SOURCE_DIR}/src/3rdparty/eos/3rdparty/glm-0.9.7.0")

  ## #################################################################
  ## Dependencies - Boost
  ## #################################################################
  # eos needs these
  hunter_add_package(Boost COMPONENTS system filesystem)
  find_package(Boost CONFIG REQUIRED system filesystem)

  set(EOS_DEPS Boost::system Boost::filesystem)
else()
  add_definitions(-DFACELAB_USE_EOS=0)
endif()

## #################################################################
## Dependencies - dest
## #################################################################
hunter_add_package(dest)
find_package(dest CONFIG REQUIRED)
add_definitions(-DDEST_WITH_OPENCV=1)

## #################################################################
## Dependencies - OpenCV
## #################################################################
hunter_add_package(OpenCV)
find_package(OpenCV REQUIRED)

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
