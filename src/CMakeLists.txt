#include <facelab/homomorphic.hpp>
include_directories(lib)
include_directories("${CMAKE_SOURCE_DIR}/src/3rdparty/local_laplacian_filters")
add_subdirectory(3rdparty)
add_subdirectory(lib)
add_subdirectory(app)
