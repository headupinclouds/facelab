# Copyright (c) 2015, David Hirvonen
# All rights reserved.

include_directories(lib)

add_definitions(-DDEST_WITH_OPENCV=1)

# DEST dependencies
include_directories("${CMAKE_SOURCE_DIR}/src/3rdparty/dummy")
include_directories("${CMAKE_SOURCE_DIR}/src/3rdparty/dest/ext")
include_directories("${CMAKE_SOURCE_DIR}/src/3rdparty/dest/inc")

# LLF dependencies
include_directories("${CMAKE_SOURCE_DIR}/src/3rdparty/local_laplacian_filters")

# EOS dependencies
include_directories("${CMAKE_SOURCE_DIR}/src/3rdparty/eos/include")
include_directories("${CMAKE_SOURCE_DIR}/src/3rdparty/eos/3rdparty/glm-0.9.7.0")
add_subdirectory(3rdparty)

add_subdirectory(lib)

add_subdirectory(app)

