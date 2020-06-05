ExternalProject_Add(googletest
PREFIX "${CMAKE_BINARY_DIR}/lib"
GIT_REPOSITORY "https://github.com/google/googletest.git"
GIT_TAG "master"
CMAKE_ARGS 
    "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/lib/installed" 
    "-DCMAKE_BUILD_TYPE=Debug"
    "-DCMAKE_CXX_COMPILER=${CXX}"
    "-DCMAKE_C_COMPILER=${CC}"
)


#link_directories(${CMAKE_BINARY_DIR}/lib/installed/lib)
#message("google test lib dir: ${CMAKE_BINARY_DIR}/lib/installed/lib")