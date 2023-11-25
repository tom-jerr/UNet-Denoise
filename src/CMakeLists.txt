set(OpenCV_DIR "C:/opencv/build")
set(Torch_DIR "C:/Libtorch/libtorch/share/cmake/Torch")
# find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(DNOISE Denoise.cpp)
target_include_directories(DNOISE PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DNOISE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
