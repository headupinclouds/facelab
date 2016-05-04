## #################################################################
## Executable
## #################################################################

add_executable(filter filter.cpp FaceLandmarker.h FaceLandmarker.cpp)
target_link_libraries(filter facelab llf ${OpenCV_LIBS} Eigen::eigen dest Boost::system Boost::filesystem)
install(TARGETS filter DESTINATION bin)
set_property(TARGET filter PROPERTY FOLDER "apps")

