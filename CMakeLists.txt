cmake_minimum_required(VERSION 3.1)
project(xevo)

# set versioning
set(XEVO_VERSION_MAJOR "0" CACHE STRING "Major version")
set(XEVO_VERSION_MINOR "0" CACHE STRING "Minor version")
set(XEVO_VERSION_PATCH "2" CACHE STRING "Patch number")
set(XEVO_VERSION_TWEAK "0" CACHE STRING "tweak number")
set(XEVO_VERSION_STRING "${XEVO_VERSION_MAJOR}"
								        ".${XEVO_VERSION_MINOR}"
								        ".${XEVO_VERSION_PATCH}"
								        ".${XEVO_VERSION_TWEAK}")
string(REPLACE ";" "" XEVO_VERSION_STRING "${XEVO_VERSION_STRING}")
message(STATUS "XEVO version ${XEVO_VERSION_STRING}")

option(ENABLE_THREADS "Enable multi-threading" ON) # Enabled by default
option(INSTALL_LIB "Install xevo" ON)
option(BUILD_TESTS "Build tests" OFF)
# add a target to generate API documentation with Doxygen
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" OFF)

set(XEVO_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(BUILD_DOCUMENTATION)
  add_subdirectory(doc)
endif(BUILD_DOCUMENTATION)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

if(ENABLE_THREADS AND BUILD_TESTS)
	find_package(GTest REQUIRED)
  find_package(Threads REQUIRED)
endif(ENABLE_THREADS AND BUILD_TESTS)

find_package(xtensor REQUIRED)

add_definitions(-DXTENSOR_ENABLE_XSIMD)
add_definitions(-DXTENSOR_USE_XSIMD)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx2 -ffast-math")

set(xevo_INCLUDE_DIRS ${xevo_SOURCE_DIR}/include)

message("xtensor include dir: ${xtensor_INCLUDE_DIRS}")
message("xevo include dir: ${xevo_INCLUDE_DIRS}")


#set(XKRIGE_SOURCES src/xtensor_example.cpp)

set(XEVO_SOURCES_TEST test/unittest_main.cpp
											test/test_functors.cpp
											test/test_ga.cpp
											test/test_pso.cpp)

set(XEVO_HEADERS ${XEVO_INCLUDE}/xevo/ga.hpp
                 ${XEVO_INCLUDE}/xevo/pso.hpp
								 ${XEVO_INCLUDE}/xevo/pso_ga.hpp
								 ${XEVO_INCLUDE}/xevo/functors.hpp
								 ${XEVO_INCLUDE}/xevo/analytical_functions.hpp)

add_library(xevo INTERFACE)

target_include_directories(xevo INTERFACE
    $<BUILD_INTERFACE:${XEVO_INCLUDE}>
    $<INSTALL_INTERFACE:include>)

target_compile_features(xevo INTERFACE cxx_std_14)

# Install XEVO
# ============
if(INSTALL_LIB)
	include(GNUInstallDirs)
	include(CMakePackageConfigHelpers)

	install(TARGETS xevo
					EXPORT xevo-targets)

	# Makes the project importable from the build directory
	export(EXPORT xevo-targets
				FILE "${CMAKE_CURRENT_BINARY_DIR}/xevoTargets.cmake")

	install(FILES ${XEVO_HEADERS}
					DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xevo)

	set(XEVO_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/xevo" CACHE
			STRING "install path for xevoConfig.cmake")

	configure_package_config_file(xevoConfig.cmake.in
																"${CMAKE_CURRENT_BINARY_DIR}/xevoConfig.cmake"
																INSTALL_DESTINATION ${XEVO_CMAKECONFIG_INSTALL_DIR})
	# xevo is header-only and does not depend on the architecture.
	# Remove CMAKE_SIZEOF_VOID_P from xevoConfigVersion.cmake so that an xevoConfig.cmake
	# generated for a 64 bit target can be used for 32 bit targets and vice versa.
	set(_XEVO_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
	unset(CMAKE_SIZEOF_VOID_P)
	write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
																	VERSION ${XEVO_VERSION_STRING}
																	COMPATIBILITY AnyNewerVersion)
	set(CMAKE_SIZEOF_VOID_P ${_XEVO_CMAKE_SIZEOF_VOID_P})
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xevoConfig.cmake
								${CMAKE_CURRENT_BINARY_DIR}/xevoConfigVersion.cmake
					DESTINATION ${XEVO_CMAKECONFIG_INSTALL_DIR})
	install(EXPORT xevo-targets
					FILE xevoTargets.cmake
					DESTINATION ${XEVO_CMAKECONFIG_INSTALL_DIR})
endif(INSTALL_LIB)

if(BUILD_TESTS)
 ADD_DEFINITIONS(-DTEST_XEVO)
 add_executable(xevo_tests ${XEVO_SOURCES_TEST} ${XEVO_INCLUDE})
 target_include_directories(xevo_tests PRIVATE ${xevo_INCLUDE_DIRS} 
                                               ${xtensor_INCLUDE_DIRS}
                                               ${GTEST_INCLUDE_DIRS})

 target_link_libraries(xevo_tests GTest::GTest GTest::Main)
endif(BUILD_TESTS)

if(BUILD_TESTS AND MSVC)
 target_compile_options(xevo_tests PRIVATE /EHsc /MP /bigobj)
 set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif(BUILD_TESTS AND MSVC)