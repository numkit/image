cmake_minimum_required(VERSION 3.20)
project(numkit_image CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUMKIT_WITH_SIMD "Enable Highway SIMD acceleration" ON)

include(FetchContent)

# Dependency: numkit/value
if(NOT TARGET numkit_value)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../value/CMakeLists.txt")
        message(STATUS "Using local numkit/value from ${CMAKE_CURRENT_SOURCE_DIR}/../value")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../value" _deps/numkit_value)
    else()
        FetchContent_Declare(
            numkit_value
            GIT_REPOSITORY https://github.com/numkit/value.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_value)
    endif()
endif()

# Dependency: numkit/sys
if(NOT TARGET numkit_sys)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../sys/CMakeLists.txt")
        message(STATUS "Using local numkit/sys from ${CMAKE_CURRENT_SOURCE_DIR}/../sys")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../sys" _deps/numkit_sys)
    else()
        FetchContent_Declare(
            numkit_sys
            GIT_REPOSITORY https://github.com/numkit/sys.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_sys)
    endif()
endif()

# Dependency: numkit/ops
if(NOT TARGET numkit_ops)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ops/CMakeLists.txt")
        message(STATUS "Using local numkit/ops from ${CMAKE_CURRENT_SOURCE_DIR}/../ops")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ops" _deps/numkit_ops)
    else()
        FetchContent_Declare(
            numkit_ops
            GIT_REPOSITORY https://github.com/numkit/ops.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_ops)
    endif()
endif()

# Dependency: numkit/signal
if(NOT TARGET numkit_signal_core)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../signal/CMakeLists.txt")
        message(STATUS "Using local numkit/signal from ${CMAKE_CURRENT_SOURCE_DIR}/../signal")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../signal" _deps/numkit_signal)
    else()
        FetchContent_Declare(
            numkit_signal
            GIT_REPOSITORY https://github.com/numkit/signal.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_signal)
    endif()
endif()

file(GLOB_RECURSE IMAGE_COMPUTE_SOURCES
    "src/arithmetic/*.cpp"
    "src/color/*.cpp"
    "src/contrast/*.cpp"
    "src/filter/*.cpp"
    "src/geom/*.cpp"
    "src/morph/*.cpp"
    "src/object/*.cpp"
    "src/quality/*.cpp"
    "src/region/*.cpp"
    "src/segment/*.cpp"
    "src/texture/*.cpp"
    "src/transform/*.cpp"
    "src/type_convert/*.cpp"
)

file(GLOB_RECURSE IMAGE_PLUGIN_SOURCES
    "src/register/*.cpp"
    "src/module.cpp"
)

add_library(numkit_image_core STATIC ${IMAGE_COMPUTE_SOURCES})
add_library(numkit::image_core ALIAS numkit_image_core)
set_target_properties(numkit_image_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(numkit_image_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_image_core PUBLIC numkit_value numkit_sys numkit_ops numkit_signal_core)

if(NUMKIT_WITH_SIMD)
    target_compile_definitions(numkit_image_core PUBLIC NUMKIT_WITH_SIMD=1)
    target_link_libraries(numkit_image_core PRIVATE hwy hwy_contrib)
endif()

if(MSVC)
    target_compile_options(numkit_image_core PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_image_core PRIVATE -Wall -Wextra -fPIC)
endif()

# Shared plugin module
add_library(numkit_image SHARED ${IMAGE_PLUGIN_SOURCES})
add_library(numkit::image ALIAS numkit_image)

target_compile_definitions(numkit_image PRIVATE NUMKIT_PLUGIN_BUILD=1)

target_include_directories(numkit_image
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_image PUBLIC numkit_image_core)

if(MSVC)
    target_compile_options(numkit_image PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_image PRIVATE -Wall -Wextra -fPIC)
endif()
