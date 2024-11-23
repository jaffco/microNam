cmake_minimum_required(VERSION 3.10)

# Make sure this matches ./NAM/version.h!
project(NAM VERSION 0.2.0)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++20 -Os -ffunction-sections -fdata-sections")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --gc-sections")

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    include_directories(SYSTEM /usr/local/include)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-dead_strip")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --gc-sections")
    link_libraries(stdc++fs)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    message(FATAL_ERROR "Unrecognized Platform!")
endif()

set(NAM_DEPS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies")

add_subdirectory(tools)