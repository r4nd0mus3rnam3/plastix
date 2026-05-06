add_executable(deep_mlp_classifier deep_mlp_classifier.cpp)
target_link_libraries(deep_mlp_classifier PRIVATE plastix)
target_compile_options(deep_mlp_classifier PRIVATE -Wall -Wextra -Wpedantic)
