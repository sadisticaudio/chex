cmake_minimum_required(VERSION 3.15...3.26)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE on)
set(_GLIBCXX_HAVE_STACKTRACE 1)
set(theTorchRoot "/media/frye/sda5/anaconda3/envs/pytorch_env/lib/python3.10/site-packages/torch")
message("home for Torch = " ${theTorchRoot})

set(kineto_LIBRARY ${theTorchRoot}/lib/libkineto.a)

set(CUDA_HOME /usr/local/cuda)
set(CUDNN_ROOT_DIR ${CUDA_HOME}/lib64)
set(_GLIBCXX_USE_CXX11_ABI 0)
set(CUDACXX ${CUDA_HOME}/bin/nvcc)
set(CMAKE_CUDA_COMPILER ${CUDA_HOME}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CAFFE2_USE_CUDNN 1)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_HOME})
set(CUDA_LAUNCH_BLOCKING 0)
set(TORCH_USE_CUDA_DSA 0)
set(Torch_INCLUDE_DIR "${theTorchRoot}/include;${theTorchRoot}/include/torch/csrc/api/include")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${theTorchRoot})#/libtorch;/home/frye/CODE/lib/eigen-3.4.0;/home/frye/CODE/lib/essentia")# /home/frye/CODE/lib/eigen-3.4.0 /home/frye/CODE/lib/essentia)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(BOOST_LOCALE_HIDE_AUTO_PTR 1)

set(EINOPS_HOME "/media/frye/sda5/CODE/einops-cpp")
add_compile_definitions(EINOPS_TORCH_BACKEND)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-std=c++2b ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -allow-unsupported-compiler")
set(CXX_STANDARD_REQUIRED ON)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(Torch_DIR ${theTorchRoot}/share/cmake/Torch/)
find_package(Torch REQUIRED CONFIG)
set(Boost_DIR "/media/frye/sda5/boost_1_85_0/stage/lib/cmake/Boost-1.85.0")
find_package(Boost REQUIRED COMPONENTS python)

# -Wno-psabi
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -v -fPIC -fno-omit-frame-pointer -pthread -Wno-deprecated-declarations")
message(${CMAKE_CXX_FLAGS})
# set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb3 -rdynamic -O0 -fsanitize=address")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0 -fpermissive -ggdb3 -rdynamic -fsanitize=address -fno-eliminate-unused-debug-symbols -fno-eliminate-unused-debug-types -fsanitize=undefined -fsanitize=float-divide-by-zero -Wpedantic -Wuninitialized")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2")
# -ftime-report -fsanitize=address -fconstexpr-steps=12712400 -stdlib=libstdc++ -stdlib=libc++ -v -fcompare-debug-second  -stdlib=libc++ -L/usr/lib/llvm-14/lib -I/usr/lib/llvm-14/include/c++/v1/

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "../lib")

# list(APPEND PyVersions 3.8 3.9 3.10 3.11 3.12)
# set(myTargets "")

set(pyVer ${PYVERSION})

# foreach(pyVer IN LISTS PyVersions)

string(REPLACE "." "" pyName ${pyVer})
# set(myTarget Chex${pyName})
# list(APPEND myTargets ${myTarget})

set(pythonPrefix "/media/frye/sda5/anaconda3/envs/python${pyVer}") #set(pythonPrefix "/usr")
set(Python_ROOT_DIR "${pythonPrefix}")
set(Python_EXECUTABLE "${pythonPrefix}/bin/python")
set(Python_LIBRARY "${pythonPrefix}/lib/python${pyVer}")
set(Python_INCLUDE_DIR "${pythonPrefix}/include/python${pyVer}")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Target: chex${pyName}, Prefix: ${pythonPrefix}")
message(STATUS "Boost_LIBRARY_DIRS: chex${Boost_LIBRARY_DIRS}, Boost_LIBRARIES: ${Boost_LIBRARIES}")
message(STATUS "Boost_LIB_VERSION: chex${Boost_LIB_VERSION}, Boost_Python_LIBRARY: ${Boost_Python_LIBRARY}")

add_library(Chex${pyName} STATIC src/Chex.cpp)
target_compile_features(Chex${pyName} PUBLIC cxx_std_23)

target_link_libraries(Chex${pyName} PRIVATE
  -L${theTorchRoot}/lib -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda
  -L/usr/lib/x86_64-linux-gnu -lcuda -L/usr/local/cuda-11.8/lib64 -lnvrtc -lnvToolsExt -lcudart)

target_include_directories(Chex${pyName} PUBLIC ${Torch_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${Python_INCLUDE_DIR} ${CUDA_HOME}/include ${EINOPS_HOME}/include)
# add_library(source-combined STATIC "")
# set_target_properties(source-combined PROPERTIES LINKER_LANGUAGE CXX)
# target_link_libraries(source-combined PUBLIC
#   -Wl,--whole-archive
#   $<TARGET_PROPERTY:Chex${pyName},INTERFACE_LINK_LIBRARIES>
#   -Wl,--no-whole-archive
# )
python_add_library(chex MODULE src/chex.cpp WITH_SOABI)
install(TARGETS chex DESTINATION .)
set_target_properties(chex PROPERTIES PREFIX "")

target_link_libraries(chex PRIVATE
  -Wl,--whole-archive
  Chex${pyName}
  -Wl,--no-whole-archive
  -Wl,--whole-archive
  -L${Boost_LIBRARIES_DIR} -lboost_python${pyName} -lboost_system
  -Wl,--no-whole-archive
  )

target_include_directories(chex PRIVATE ${Boost_INCLUDE_DIRS} ${Python_INCLUDE_DIR})

  # endforeach()

# message(STATUS "list of targets= ${myTargets}")

# clang -std=c++20 -std=gnu++20 -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /media/frye/sda5/libtorch-2.0.1-RC/include -isystem /media/frye/sda5/libtorch-2.0.1-RC/include/torch/csrc/api/include -isystem /media/frye/sda5/boost_1_85_0 -isystem /usr/include/python3.10 -c -o m.o Chex.cpp
# clang -o m.so m.o -shared -L/media/frye/sda5/libtorch-2.0.1-RC/lib -l torch -Wl,-rpath /media/frye/sda5/libtorch-2.0.1-RC/lib -L/usr/lib/x86_64-linux-gnu -l cuda -Wl,-rpath /usr/lib/x86_64-linux-gnu -L/usr/local/cuda-11.8/lib64 -l nvrtc -l nvToolsExt -l cudart -Wl,-rpath /usr/local/cuda-11.8/lib64 -L/media/frye/sda5/boost_1_85_0/stage/lib -l boost_python310 -l boost_system -Wl,-rpath /media/frye/sda5/boost_1_85_0/stage/lib

add_executable(cpptest cpptest.cpp)
set_target_properties(cpptest PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_link_libraries(cpptest PRIVATE Chex310 -L${Boost_LIBRARIES_DIR} -lboost_python310 -lboost_system)
target_include_directories(cpptest PUBLIC ${Torch_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${Python_INCLUDE_DIR} ${CUDA_HOME}/include ${EINOPS_HOME}/include)