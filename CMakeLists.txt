cmake_minimum_required(VERSION 3.15)
project(edge_rag C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

# Strict warnings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -O3")

# Enable NEON if available (can be toggled by a toolchain file)
# add_compile_options(-mfpu=neon)

include_directories(include)

# Main executable
add_executable(edge_rag src/main.c src/memory.c)

# Add math library and pthreads
target_link_libraries(edge_rag m pthread)
