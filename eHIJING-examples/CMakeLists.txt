cmake_minimum_required(VERSION 2.8)
project(eHIJING CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmakemodules")
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" CXX17)
if(NOT CXX17)
  message(FATAL_ERROR "Your compiler does not support C++17")
endif()

# Find and use Pythia
find_package(Pythia8)
if (${PYTHIA8_FOUND})
  include_directories(${PYTHIA8_INCLUDE_DIR})
  Message(STATUS "Pythia8 Include dir : " ${PYTHIA8_INCLUDE_DIR})
endif()

find_package(GSL)
if (${GSL_FOUND})
  include_directories(${GSL_INCLUDE_DIR})
  Message(STATUS "GSL Include dir : " ${GSL_INCLUDE_DIR})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local"
      CACHE PATH "Install path prefix, prepended onto install directories."
      FORCE)
endif()

foreach(App "ep" "SingleParton")
add_executable(${App} ./${App}.cpp)
target_link_libraries(${App} ${PYTHIA8_LIBRARIES} ${GSL_LIBRARIES} -pthread -ldl)
install(TARGETS ${App} DESTINATION bin)
endforeach()
