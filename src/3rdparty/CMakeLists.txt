# Copyright (c) 2015, David Hirvonen
# All rights reserved.

### Local laplacian filters 
set(SKIP_INSTALL ON CACHE BOOL "Avoid installation of submodule libraries" )
add_subdirectory(local_laplacian_filters)
unset(SKIP_INSTALL)

set_property(TARGET llf PROPERTY FOLDER "libs/3rdparty/llf")
set_property(TARGET locallaplacian PROPERTY FOLDER "apps/3rdparty")

### DEST
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(dummy)
set(SKIP_INSTALL ON CACHE BOOL "Avoid installation of submodule libraries" )
add_subdirectory(dest)
unset(SKIP_INSTALL)

set_property(TARGET dest PROPERTY FOLDER "libs/3rdparty")

# foreach(TARGET1
#     dest_tests
#     dest_expression_transfer
#     dest_align
#     dest_track_video
#     dest_train
#     dest_face_swap
#     dest_evaluate
#     dest_gen_rects
#     dest_show_landmarks
#     )
#   set_property(TARGET ${TARGET1} PROPERTY FOLDER "apps/3rdparty")    
# endforeach(TARGET1)
