cmake_minimum_required(VERSION 3.20)
project(numkit_image CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUMKIT_WITH_SIMD "Enable Highway SIMD acceleration" ON)
option(NUMKIT_BUILD_TESTS "Build tests" OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_NUMKIT_TOP_BUILD_TESTS ${NUMKIT_BUILD_TESTS})
    set(NUMKIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
endif()

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
    if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../signal/build/Release/numkit_signal_core.lib")
        add_library(numkit_signal_core STATIC IMPORTED)
        set_target_properties(numkit_signal_core PROPERTIES
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../signal/build/Release/numkit_signal_core.lib"
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../signal/include"
            INTERFACE_LINK_LIBRARIES "numkit_value;numkit_sys;numkit_ops"
        )
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../signal/CMakeLists.txt")
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

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(NUMKIT_BUILD_TESTS ${_NUMKIT_TOP_BUILD_TESTS} CACHE BOOL "" FORCE)
endif()

if(NUMKIT_BUILD_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    if(NOT TARGET numkit)
        if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/Release/numkit.lib")
            add_library(numkit STATIC IMPORTED)
            set_target_properties(numkit PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/Release/numkit.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/include"
                INTERFACE_COMPILE_DEFINITIONS "NUMKIT_CORE_BUILD=1"
                INTERFACE_LINK_LIBRARIES "numkit_value;numkit_sys;numkit_ops"
            )
        else()
            set(NUMKIT_BUILD_CLI OFF CACHE BOOL "" FORCE)
            set(NUMKIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/CMakeLists.txt")
                add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../numkit" _deps/numkit)
            else()
                FetchContent_Declare(
                    numkit_core
                    GIT_REPOSITORY https://github.com/numkit/numkit.git
                    GIT_TAG main
                )
                FetchContent_MakeAvailable(numkit_core)
            endif()
            set(NUMKIT_BUILD_TESTS ON CACHE BOOL "" FORCE)
        endif()
    endif()

    if(NOT TARGET GTest::gtest_main)
        if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest_main.lib")
            add_library(GTest::gtest STATIC IMPORTED)
            set_target_properties(GTest::gtest PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src/googletest/include"
            )
            add_library(GTest::gtest_main STATIC IMPORTED)
            set_target_properties(GTest::gtest_main PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest_main.lib"
                INTERFACE_LINK_LIBRARIES "GTest::gtest"
            )
        else()
            set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
            set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
            set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src/CMakeLists.txt")
                set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src" CACHE PATH "" FORCE)
            endif()
            FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v1.14.0
            )
            FetchContent_MakeAvailable(googletest)
        endif()
    endif()

    file(GLOB IMAGE_TEST_SOURCES "tests/*_test.cpp")
    add_executable(numkit_image_test ${IMAGE_TEST_SOURCES})
    target_include_directories(numkit_image_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${numkit_ops_SOURCE_DIR}/src/image_io
    )
    target_link_libraries(numkit_image_test PRIVATE
        numkit_image_core
        numkit
        GTest::gtest_main
    )
    if(NUMKIT_WITH_SIMD)
        target_link_libraries(numkit_image_test PRIVATE hwy hwy_contrib)
    endif()
    add_dependencies(numkit_image_test numkit_image)
    if(MSVC)
        target_compile_options(numkit_image_test PRIVATE /W3 /MP /bigobj)
    endif()
    add_test(NAME numkit_image_test COMMAND numkit_image_test)
endif()
