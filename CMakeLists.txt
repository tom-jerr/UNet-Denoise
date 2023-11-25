cmake_minimum_required(VERSION 3.12)
project(UnetDenoiseVSCode)
# ------ Standard ------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_BUILD_TYPE "Release")
include_directories(include)
add_subdirectory(test)
add_subdirectory(src)

