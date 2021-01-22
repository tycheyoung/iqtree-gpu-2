#ifndef KERNELS__CUH__
#define KERNELS__CUH__

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/cuda_runtime.h"
#include "cuda_runtime_api.h"

#define USE_FLOAT
#define _MAX(a,b)  fmaxf(a,b)

#define N_THREADS 1024

#ifdef USE_FLOAT
typedef float elem_t;
#else
typedef double elem_t;
#endif

typedef struct
{
    elem_t A;
    elem_t C;
    elem_t G;
    elem_t T;
} nodeLikelihood;


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

elem_t cuda_maxll_score(char* seqs, int* treeArray, elem_t* treeLengthArray, int* node_level, elem_t* rate_mat, 
                        elem_t* pi, int tree_total_node_num, int seq_length, int seq_num);

#endif
