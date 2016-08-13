### Emulate toolchain
set(CMAKE_OSX_SYSROOT "iphoneos")
set(CMAKE_XCODE_EFFECTIVE_PLATFORMS "-iphoneos;-iphonesimulator")
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
### -- end

#include("$ENV{SUGAR_ROOT}/cmake/Sugar")
include(sugar_include)

include_directories("${SUGAR_ROOT}/examples/third_party")

sugar_include("./sources")
sugar_include("${SUGAR_ROOT}/examples/resources/ios/icons")
sugar_include("${SUGAR_ROOT}/examples/resources/ios/images")

if(NOT XCODE_VERSION)
  sugar_fatal_error("Xcode only")
endif()

add_executable(filter ${SOURCES} ${DEFAULT_IOS_IMAGES} ${IOS_ICONS})

set_target_properties(
  filter
  PROPERTIES
  MACOSX_BUNDLE YES
  MACOSX_BUNDLE_INFO_PLIST
  "${SUGAR_ROOT}/examples/plist/ios/empty.plist.in"
  XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
  XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2" # Universal (iPad + iPhone)
  # http://stackoverflow.com/a/20982506/2288008
  XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
  XCODE_ATTRIBUTE_INSTALL_PATH "${CMAKE_BINARY_DIR}/ProductRelease"
  # By default this setting is empty =>
  # archives not showing up in Xcode organizer.
  # http://stackoverflow.com/a/8102602/2288008
  XCODE_ATTRIBUTE_COMBINE_HIDPI_IMAGES "NO" # If this setting is "YES"
  # application can't pass archive validation.
  # http://stackoverflow.com/a/24040412/2288008
  RESOURCE "${DEFAULT_IOS_IMAGES};${IOS_ICONS}"
  )

set_target_properties(
  filter
  PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_NAME
  "Filter"
  XCODE_ATTRIBUTE_BUNDLE_IDENTIFIER
  "com.elucideye.filter"
  )

set_target_properties(
  filter
  PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_NAME[variant=Debug]
  "Filter-Dbg"
  XCODE_ATTRIBUTE_BUNDLE_IDENTIFIER[variant=Debug]
  "com.elucideye.filter.debug"  
  )

target_link_libraries(
  filter
  "-framework CoreData"
  "-framework CoreGraphics"
  "-framework Foundation"
  "-framework UIKit"
  )
