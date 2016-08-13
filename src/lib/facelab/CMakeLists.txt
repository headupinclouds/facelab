# Copyright (c) 2015, David Hirvonen
# All rights reserved.

if(FACELAB_USE_EOS)
  # 3D mesh fitter
  set(EOS_SRCS
    FaceLandmarkMeshMapper.cpp
    FaceLandmarkMeshMapper.h
    )
endif()

add_library(facelab

  ${EOS_SRCS}

  # CPR face landmark interface
  FaceLandmarker.cpp  
  FaceLandmarker.h

  # Filtering API
  Filter.h

  # homomorphic filter
  homomorphic.cpp
  homomorphic.hpp
  shift.hpp
  )

set_property(TARGET facelab PROPERTY FOLDER "libs")

message("OpenCV_LIBS ${OpenCV_LIBS}")
message("===========================")
message("===========================")
message("===========================")
message("===========================")

target_link_libraries(facelab ${OpenCV_LIBS} dest::dest)

if(FACELAB_USE_EOS)
  target_link_libraries(facelab Boost::system Boost::filesystem)
endif()

### Install ###
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(config_install_dir "lib/cmake/${PROJECT_NAME}")
set(include_install_dir "include")
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${version_config}" COMPATIBILITY SameMajorVersion
)

# Note: use 'targets_export_name'
configure_file("../../../cmake/Config.cmake.in" "${project_config}" @ONLY)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT "${targets_export_name}"
  LIBRARY DESTINATION "lib"
  ARCHIVE DESTINATION "lib"
  RUNTIME DESTINATION "bin"
  INCLUDES DESTINATION "${include_install_dir}/${PROJECT_NAME}"
)

install(
  FILES
  homomorphic.hpp
  DESTINATION "${include_install_dir}/${PROJECT_NAME}"
)

install(
  FILES "${project_config}" "${version_config}"
  DESTINATION "${config_install_dir}"
)

install(
  EXPORT "${targets_export_name}"
  NAMESPACE "${namespace}"
  DESTINATION "${config_install_dir}"
)
