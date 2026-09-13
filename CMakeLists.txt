cmake_minimum_required(VERSION 3.15)
project(edge_rag C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

# Strict warnings
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -O3")
elseif(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /O2")
endif()

# Enable NEON if available (can be toggled by a toolchain file)
# add_compile_options(-mfpu=neon)

include_directories(include)

# Main executable
add_executable(edge_rag src/main.c src/memory.c src/math.c src/mmap_utils.c src/loader.c)

# Add math library and pthreads
if(UNIX)
    target_link_libraries(edge_rag m pthread)
endif()

# Benchmark executable
add_executable(benchmark tests/benchmarks.c src/math.c src/memory.c src/mmap_utils.c src/loader.c)
if(UNIX)
    target_link_libraries(benchmark m pthread)
endif()
