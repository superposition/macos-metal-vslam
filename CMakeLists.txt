cmake_minimum_required(VERSION 3.28)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

project(macos-metal-vslam VERSION 0.1.0 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "Build for Apple Silicon" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(APPLE AND EXISTS "/opt/homebrew/opt/opencv")
  list(PREPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/opencv")
endif()

if(APPLE AND EXISTS "/opt/homebrew/anaconda3/lib/cmake/Boost-1.82.0")
  set(Boost_DIR "/opt/homebrew/anaconda3/lib/cmake/Boost-1.82.0")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/.deps/gtsam-install")
  list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/.deps/gtsam-install")
endif()

find_package(OpenCV REQUIRED COMPONENTS calib3d core features2d imgproc video videoio)
find_package(GTSAM CONFIG REQUIRED)
find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

set(SHADER_SOURCE "${CMAKE_SOURCE_DIR}/shaders/luma.metal")
set(SHADER_AIR "${CMAKE_BINARY_DIR}/luma.air")
set(SHADER_LIB "${CMAKE_BINARY_DIR}/luma.metallib")

add_custom_command(
  OUTPUT "${SHADER_AIR}"
  COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -c "${SHADER_SOURCE}" -o "${SHADER_AIR}"
  DEPENDS "${SHADER_SOURCE}"
  VERBATIM
)

add_custom_command(
  OUTPUT "${SHADER_LIB}"
  COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib "${SHADER_AIR}" -o "${SHADER_LIB}"
  DEPENDS "${SHADER_AIR}"
  VERBATIM
)

add_custom_target(vslam_shaders ALL DEPENDS "${SHADER_LIB}")

add_executable(macos-metal-vslam
  src/AppMain.mm
  src/MetalLumaProcessor.mm
  src/MonocularTracker.cpp
)

target_include_directories(macos-metal-vslam PRIVATE include)
target_link_libraries(macos-metal-vslam PRIVATE
  ${OpenCV_LIBS}
  gtsam
  "-framework AppKit"
  "-framework Foundation"
  "-framework Metal"
  "-framework SceneKit"
)

target_compile_options(macos-metal-vslam PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall -Wextra -Wpedantic>
)

set_source_files_properties(
  src/AppMain.mm
  src/MetalLumaProcessor.mm
  PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

set_target_properties(macos-metal-vslam PROPERTIES
  BUILD_RPATH "${CMAKE_SOURCE_DIR}/.deps/gtsam-install/lib;/opt/homebrew/anaconda3/lib"
)

add_dependencies(macos-metal-vslam vslam_shaders)

add_custom_command(
  TARGET macos-metal-vslam
  POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${SHADER_LIB}"
          "$<TARGET_FILE_DIR:macos-metal-vslam>/luma.metallib"
)
