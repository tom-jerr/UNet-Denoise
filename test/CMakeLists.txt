add_executable(DNOISETEST DenoiseTest.cpp)
target_include_directories(DNOISETEST PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})
target_link_libraries(DNOISETEST  DNOISE ${TORCH_LIBRARIES} ${OpenCV_LIBS} pthread )