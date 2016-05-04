## #################################################################
## Executable
## #################################################################

add_executable(retina retina.cpp)
target_link_libraries(retina facelab llf ${OpenCV_LIBS} Eigen::eigen dest)
install(TARGETS retina DESTINATION bin)

set_property(TARGET retina PROPERTY FOLDER "apps")