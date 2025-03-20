cmake_minimum_required(VERSION 3.19.1)

project(Ancalagon CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(ARGUMENTUM_BUILD_STATIC_LIBS ON)

add_subdirectory(external/argumentum)
include_directories(${PROJECT_SOURCE_DIR}/headers 
                    ${PROJECT_SOURCE_DIR}/external/argumentum/include)

file(GLOB sources ${PROJECT_SOURCE_DIR}/sources/*.cu
                  ${PROJECT_SOURCE_DIR}/sources/*.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/headers/*.h)

add_executable(Ancalagon ${headers} ${sources})

target_link_libraries(Ancalagon gmp gmpxx Argumentum::argumentum)

set_target_properties(Ancalagon PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
