find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
add_executable(DNOISE Denoise.cpp)
target_link_libraries(DNOISE ${TORCH_LIBRARIES})
