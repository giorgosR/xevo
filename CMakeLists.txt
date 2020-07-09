cmake_minimum_required(VERSION 3.1)
project(xnio)

# set versioning
set(XNIO_VERSION_MAJOR "0" CACHE STRING "Major version")
set(XNIO_VERSION_MINOR "0" CACHE STRING "Minor version")
set(XNIO_VERSION_PATCH "1" CACHE STRING "Patch number")
set(XNIO_VERSION_TWEAK "0" CACHE STRING "tweak number")
set(XNIO_VERSION_STRING "${XNIO_VERSION_MAJOR}"
								        ".${XNIO_VERSION_MINOR}"
								        ".${XNIO_VERSION_PATCH}"
								        ".${XNIO_VERSION_TWEAK}")
string(REPLACE ";" "" XNIO_VERSION_STRING "${XNIO_VERSION_STRING}")
message(STATUS "XNIO version ${XNIO_VERSION_STRING}")

option(ENABLE_THREADS "Enable multi-threading" OFF) # Enabled by default
option(INSTALL_LIB "Install xnio" ON)
option(BUILD_TESTS "Build tests" OFF)
# add a target to generate API documentation with Doxygen
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" OFF)

set(XNIO_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(BUILD_DOCUMENTATION)
  add_subdirectory(doc)
endif(BUILD_DOCUMENTATION)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

if(ENABLE_THREADS AND BUILD_TESTS)
  find_package(Threads REQUIRED)
  add_definitions(-DENABLE_THREADS)
  message(STATUS "Threads enabled.")
endif(ENABLE_THREADS AND BUILD_TESTS)


find_package(xtensor REQUIRED)

add_definitions(-DXTENSOR_ENABLE_XSIMD)
add_definitions(-DXTENSOR_USE_XSIMD)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx2 -ffast-math")

set(xnio_INCLUDE_DIRS ${xnio_SOURCE_DIR}/include)

message("xtensor include dir: ${xtensor_INCLUDE_DIRS}")
message("xnio include dir: ${xnio_INCLUDE_DIRS}")


#set(XKRIGE_SOURCES src/xtensor_example.cpp)

set(XNIO_SOURCES_TEST test/unittest_main.cpp
											test/test_functors.cpp
											test/test_ga.cpp)

set(XNIO_HEADERS ${XNIO_INCLUDE}/xnio/ga.hpp
								 ${XNIO_INCLUDE}/xnio/functors.hpp
								 ${XNIO_INCLUDE}/xnio/analytical_functions.hpp)

# if(INSTALL_LIB)
#   FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/include/xnio/*")
#   INSTALL(FILES ${files} DESTINATION xnio)
#   set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/lib/installed" CACHE PATH "Install directory.")
# endif(INSTALL_LIB)

add_library(xnio INTERFACE)

target_include_directories(xnio INTERFACE
    $<BUILD_INTERFACE:${XNIO_INCLUDE}>
    $<INSTALL_INTERFACE:include>)

target_compile_features(xnio INTERFACE cxx_std_14)

#target_link_libraries(xnio INTERFACE xtl)

# Install XNIO
# ============
if(INSTALL_LIB)
	include(GNUInstallDirs)
	include(CMakePackageConfigHelpers)

	install(TARGETS xnio
					EXPORT xnio-targets)

	# Makes the project importable from the build directory
	export(EXPORT xnio-targets
				FILE "${CMAKE_CURRENT_BINARY_DIR}/xnioTargets.cmake")

	install(FILES ${XNIO_HEADERS}
					DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xnio)

	set(XNIO_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/xnio" CACHE
			STRING "install path for xnioConfig.cmake")

	configure_package_config_file(xnioConfig.cmake.in
																"${CMAKE_CURRENT_BINARY_DIR}/xnioConfig.cmake"
																INSTALL_DESTINATION ${XNIO_CMAKECONFIG_INSTALL_DIR})
	# xnio is header-only and does not depend on the architecture.
	# Remove CMAKE_SIZEOF_VOID_P from xnioConfigVersion.cmake so that an xnioConfig.cmake
	# generated for a 64 bit target can be used for 32 bit targets and vice versa.
	set(_XNIO_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
	unset(CMAKE_SIZEOF_VOID_P)
	write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
																	VERSION ${XNIO_VERSION_STRING}
																	COMPATIBILITY AnyNewerVersion)
	set(CMAKE_SIZEOF_VOID_P ${_XNIO_CMAKE_SIZEOF_VOID_P})
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xnioConfig.cmake
								${CMAKE_CURRENT_BINARY_DIR}/xnioConfigVersion.cmake
					DESTINATION ${XNIO_CMAKECONFIG_INSTALL_DIR})
	install(EXPORT xnio-targets
					FILE xnioTargets.cmake
					DESTINATION ${XNIO_CMAKECONFIG_INSTALL_DIR})
endif(INSTALL_LIB)

if(BUILD_TESTS)
  include(ExternalProject)
  include(External_googletest)
  ADD_DEFINITIONS(-DTEST_XNIO)
  if(UNIX)
    set(GTEST_LIBRARIES ${CMAKE_BINARY_DIR}/lib/installed/lib/libgtestd.a
                        ${CMAKE_BINARY_DIR}/lib/installed/lib/libgtest_maind.a)
    message("GTEST_LIBS: ${GTEST_LIBRARIES}")
  else()
   set(GTEST_LIBRARIES ${CMAKE_BINARY_DIR}/lib/installed/lib/gtestd.lib
                       ${CMAKE_BINARY_DIR}/lib/installed/lib/libgtest_maind.lib)
   message("GTEST_LIBS: ${GTEST_LIBRARIES}")
  endif(UNIX)
endif(BUILD_TESTS)

if(BUILD_TESTS)
 add_executable(xnio_tests ${XNIO_SOURCES_TEST} ${XNIO_INCLUDE})
 target_include_directories(xnio_tests PUBLIC ${xnio_INCLUDE_DIRS} 
                                              ${xtensor_INCLUDE_DIRS}
                                              ${CMAKE_BINARY_DIR}/lib/installed/include)

 target_link_libraries(xnio_tests ${GTEST_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
 add_dependencies(xnio_tests googletest)
endif(BUILD_TESTS)

# if(MSVC)
#  target_compile_options(run PRIVATE /EHsc /MP /bigobj)
#  set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
# endif(MSVC)

# if(BUILD_TESTS)
#   if(UNIX)
#     target_include_directories(xnio_tests PUBLIC ${xnio_INCLUDE_DIRS} 
#                                                  ${xtensor_INCLUDE_DIRS}
#                                                  ${CMAKE_BINARY_DIR}/lib/installed/include)
    
#     target_link_libraries(xnio_tests ${GTEST_LIBRARIES}
#                                      ${CMAKE_THREAD_LIBS_INIT})
#     add_dependencies(xnio_tests googletest)
#     set_target_properties(xnio_tests PROPERTIES EXCLUDE_FROM_ALL TRUE)
#   else(WIN32)
#     target_include_directories(xnio_tests PUBLIC ${xnio_INCLUDE_DIRS} 
#                                                  ${xtensor_INCLUDE_DIRS} 
#                                                  ${CMAKE_BINARY_DIR}/lib/installed/include)
                                                       
#     target_link_libraries(xnio_tests ${GTEST_LIBRARIES}
#                                      Threads::Threads)
# 	  add_dependencies(xnio_tests googletest)
#   endif(UNIX)
# else()
#   if(UNIX)
#     # target_include_directories(run PUBLIC ${xkrige_INCLUDE_DIRS} 
#     #                                       ${xtensor_INCLUDE_DIRS} 
#     #                                       ${xtensor_blas_INCLUDE_DIRS})
#     # target_link_libraries(run ${OpenBLAS_LIBRARIES} ${LAPACK_LIBRARIES})
#   else(WIN32)
#     # target_include_directories(run PUBLIC ${xkrige_INCLUDE_DIRS} 
#     #                                       ${xtensor_INCLUDE_DIRS} 
#     #                                       ${xtensor_blas_INCLUDE_DIRS})
#     # target_link_libraries(run ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
#   endif(UNIX)
# endif(BUILD_TESTS)