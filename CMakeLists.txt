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

option(ENABLE_THREADS "Enable multi-threading" ON) # Enabled by default
option(INSTALL_LIB "Install xnio" OFF)
option(BUILD_TESTS "Build tests" OFF)
# add a target to generate API documentation with Doxygen
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" OFF)

if(BUILD_DOCUMENTATION)
  add_subdirectory(doc)
endif(BUILD_DOCUMENTATION)

set(CMAKE_CXX_STANDARD 14)
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

set(xnio_INCLUDE_DIRS ${xnio_SOURCE_DIR}/include)

message("xtensor include dir: ${xtensor_INCLUDE_DIRS}")
message("xnio include dir: ${xnio_INCLUDE_DIRS}")


#set(XKRIGE_SOURCES src/xtensor_example.cpp)

set(XNIO_SOURCES_TEST test/unittest_main.cpp
											test/test_functors.cpp
											test/test_ga.cpp)

set(XNIO_INCLUDE include/xnio/ga.hpp
                 include/xnio/functors.hpp)

# if(INSTALL_LIB)
#   FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/include/xnio/*")
#   INSTALL(FILES ${files} DESTINATION xnio)
#   set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/lib/installed" CACHE PATH "Install directory.")
# endif(INSTALL_LIB)

if(BUILD_TESTS)
	# Lib directories
	# ---------------
	# Third-party library
	# Third-party library
	include(ExternalProject)
	ExternalProject_Add(googletest
	PREFIX "${CMAKE_BINARY_DIR}/lib"
	GIT_REPOSITORY "https://github.com/google/googletest.git"
	GIT_TAG "master"
	CMAKE_ARGS 
		"-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/lib/installed" 
		"-DCMAKE_BUILD_TYPE=Debug"
		"-Dgtest_force_shared_crt=ON"
	)
	link_directories(${CMAKE_BINARY_DIR}/lib/installed/lib)
	message("google test lib dir: ${CMAKE_BINARY_DIR}/lib/installed/lib")
endif(BUILD_TESTS)

if(BUILD_TESTS)
 add_executable(xnio_tests ${XNIO_SOURCES_TEST} ${XNIO_INCLUDE})
 target_include_directories(xnio_tests PUBLIC ${xnio_INCLUDE_DIRS} 
                                              ${xtensor_INCLUDE_DIRS}
                                              ${CMAKE_BINARY_DIR}/lib/installed/include)

 target_link_libraries(xnio_tests gtestd gtest_maind ${CMAKE_THREAD_LIBS_INIT})
 add_dependencies(xnio_tests googletest)
 set_target_properties(xnio_tests PROPERTIES EXCLUDE_FROM_ALL TRUE)
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