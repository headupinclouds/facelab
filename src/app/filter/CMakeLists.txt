## #################################################################
## Executable
## #################################################################

add_executable(filter filter.cpp)
target_link_libraries(filter facelab llf ${OpenCV_LIBS})
install(TARGETS filter DESTINATION bin)
