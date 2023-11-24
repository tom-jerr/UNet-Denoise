set(OpenCV_DIR "C:/opencv/build")
set(Torch_DIR "C:/Libtorch/libtorch/share/cmake/Torch")
# find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(DNOISETEST DenoiseTest.cpp)
target_include_directories(DNOISETEST PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(DNOISETEST ${TORCH_LIBRARIES} ${OpenCV_LIBS})