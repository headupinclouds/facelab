# Copyright (c) 2015, David Hirvonen
# All rights reserved.

## #################################################################
## Executable
## #################################################################

add_executable(pose pose.cpp)
target_link_libraries(pose facelab ${OpenCV_LIBS} Eigen::eigen dest Boost::system Boost::filesystem)
install(TARGETS pose DESTINATION bin)
set_property(TARGET pose PROPERTY FOLDER "apps")
