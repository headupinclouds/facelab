# Copyright (c) 2015, David Hirvonen
# All rights reserved.

## #################################################################
## Executable
## #################################################################

add_executable(retina retina.cpp)
target_link_libraries(retina facelab ${OpenCV_LIBS} dest::dest)
install(TARGETS retina DESTINATION bin)

set_property(TARGET retina PROPERTY FOLDER "apps")