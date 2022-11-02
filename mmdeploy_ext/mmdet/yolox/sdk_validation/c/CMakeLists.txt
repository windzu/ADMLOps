cmake_minimum_required(VERSION 3.0.0)
project(test VERSION 0.1.0)

set(MMDeploy_DIR $ENV{MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy)
find_package(MMDeploy REQUIRED)
add_executable(${PROJECT_NAME} main.cpp)
mmdeploy_load_static(${PROJECT_NAME} MMDeployStaticModules)
mmdeploy_load_dynamic(${PROJECT_NAME} MMDeployDynamicModules)
target_link_libraries(${PROJECT_NAME} PRIVATE MMDeployLibs)
