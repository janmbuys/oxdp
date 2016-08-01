# CMake Configuration File for OxLM Project.
cmake_minimum_required(VERSION 2.6)
project(OxLM)

#############################################
# Directories
#############################################

# Store all binaries in bin/ directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib)

# Load additional CMake modules from cmake/ directory
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

# Include project path (src/)
include_directories(${PROJECT_SOURCE_DIR})

# Set path to 3rd party libraries
set(THIRD_PARTY ${PROJECT_SOURCE_DIR}/third_party)

#############################################
# Libraries
#############################################

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.54.0 COMPONENTS
  date_time filesystem program_options serialization thread REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
# link_directories(${Boost_LIBRARY_DIRS})
# set(link_libraries(train_cnlm ${Boost_LIBRARIES} cnlm)

# OpenMP
find_package(OpenMP QUIET)

# Cdec
find_package(Cdec QUIET)

# Git
find_package(Git REQUIRED)

set(CMAKE_BUILD_TYPE Debug)

#############################################
# Hackery courtesy of Karl Moritz
#############################################

# Overrides standard add_executable command and by default links all executables
# to boost. This will need changing only should we end up including code not
# reliant on boost. As if that was ever going to happen ...
macro (add_executable _name)
  _add_executable(${ARGV})
  if (TARGET ${_name})
    target_link_libraries(${_name} ${Boost_LIBRARIES})
  endif()
endmacro()

#############################################
# Generate both static and shared libraries.
#############################################

#now only shared libraries
macro(add_libraries _name)
  set(files ${ARGN})
  add_library(${_name} SHARED ${files})
  add_library(${_name}_static STATIC ${files})
  set_target_properties(${_name}_static PROPERTIES OUTPUT_NAME ${_name})
endmacro()

macro(target_link_libs _name)
  set(libs ${ARGN})
  target_link_libraries(${_name} ${libs})
  target_link_libraries(${_name}_static ${libs})
endmacro()

#############################################
# Compiler Settings
#############################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_subdirectory(utils)
add_subdirectory(corpus)
add_subdirectory(pyp)
add_subdirectory(lbl)
add_subdirectory(gdp)

# Third party libraries
