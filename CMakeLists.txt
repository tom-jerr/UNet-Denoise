cmake_minimum_required(VERSION 3.12)
project(UnetDenoise)
# ------ Standard ------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)
#add_subdirectory(test)
add_subdirectory(src)

