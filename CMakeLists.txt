cmake_minimum_required (VERSION 3.13) # I just set to my system ver on Homebrew
project (keyframe_reduction)
set (CMAKE_CXX_STANDARD 11) # OpenCV 4 requires >= cpp11

find_package( Boost 1.68 COMPONENTS program_options filesystem REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

find_package( OpenCV REQUIRED )

set (VERSION_MAJOR 1)
set (VERSION_MINOR 0)

add_executable(keyframe_reduction main.cpp)

target_link_libraries( keyframe_reduction LINK_PUBLIC ${Boost_LIBRARIES} ${OpenCV_LIBS} )
