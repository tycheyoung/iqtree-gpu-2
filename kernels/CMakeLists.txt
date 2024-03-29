# enable_language(CUDA)

add_library(kernels
kernels.cu
kernels.cuh
datatype.cuh
)
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(kernels PUBLIC cxx_std_14)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
# set(CMAKE_CXX_FLAGS	"${CMAKE_CXX_FLAGS}	-Wall")	
set(CMAKE_CUDA_FLAGS	"${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall -gencode arch=compute_61,code=sm_61 -rdc=true -lz -lm ")
# set(CMAKE_CUDA_FLAGS	"${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall -gencode arch=compute_75,code=sm_75 -rdc=true -lz -lm ")
set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(kernels PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(cub)
# target_include_directories(test PRIVATE ${YOUR_DIRECTORY})
