# Copyright (c) 2015, David Hirvonen
# All rights reserved.

add_subdirectory(filter)

if(FACELAB_USE_EOS)
  add_subdirectory(pose)
endif()
