cmake_minimum_required(VERSION 2.8)
project(eHIJING CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmakemodules")
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" CXX11)
if(NOT CXX11)
  message(FATAL_ERROR "Your compiler does not support C++11")
endif()

# Find and use Pythia
find_package(Pythia8)
if (${PYTHIA8_FOUND})
  include_directories(${PYTHIA8_INCLUDE_DIR})
  Message(STATUS "Pythia8 Include dir : " ${PYTHIA8_INCLUDE_DIR})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local"
      CACHE PATH "Install path prefix, prepended onto install directories."
      FORCE)
endif()

foreach(App "ep")
add_executable(${App} ./${App}.cpp)
target_link_libraries(${App} ${PYTHIA8_LIBRARIES} -ldl)
install(TARGETS ${App} DESTINATION bin)
endforeach()

