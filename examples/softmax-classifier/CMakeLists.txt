add_executable(softmax_classifier softmax_classifier.cpp)
target_link_libraries(softmax_classifier PRIVATE plastix)
target_compile_options(softmax_classifier PRIVATE -Wall -Wextra -Wpedantic)
