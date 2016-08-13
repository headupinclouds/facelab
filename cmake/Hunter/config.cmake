### OpenCV

### OpenCV
if(ANDROID)
  message("ANDROID =====================================================================")
  include(SetOpenCVCMakeArgs-android) 
  set_opencv_cmake_args_android()
  set(CMAKE_CXX_FLAGS_VAR -Wno-c++11-narrowing)  
elseif(is_ios)
  message("is_ios ======================================================================")
  include(SetOpenCVCMakeArgs-iOS) 
  set_opencv_cmake_args_ios()
  set(CMAKE_CXX_FLAGS_VAR -Wno-c++11-narrowing)  
elseif(APPLE) 
  message("APPLE =======================================================================")
  include(SetOpenCVCMakeArgs-osx)
  set_opencv_cmake_args_osx()
  set(CMAKE_CXX_FLAGS_VAR "-Wno-c++11-narrowing")  
elseif(${is_linux})
  message("is_linux =======================================================================")
  include(SetOpenCVCMakeArgs-nix) 
  set_opencv_cmake_args_nix()
  set(CMAKE_CXX_FLAGS_VAR -Wno-c++11-narrowing)
elseif(${is_msvc})
  message("is_msvc=========================================================================")
  include(SetOpenCVCMakeArgs-windows) 
  set_opencv_cmake_args_windows()
endif()

list(APPEND OPENCV_CMAKE_ARGS 
  BUILD_opencv_world=ON 
  BUILD_EIGEN=OFF
  CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS_VAR}
)

hunter_config(OpenCV VERSION ${HUNTER_OpenCV_VERSION} CMAKE_ARGS "${OPENCV_CMAKE_ARGS}")
