# Copyright (c) 2015, David Hirvonen
# All rights reserved.

## #################################################################
## Executable
## #################################################################

add_executable(filter filter.cpp)
target_link_libraries(filter facelab ${OpenCV_LIBS} dest::dest)
install(TARGETS filter DESTINATION bin)
set_property(TARGET filter PROPERTY FOLDER "apps")

