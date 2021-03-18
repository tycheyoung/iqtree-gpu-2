#ifndef DATATYPE__CUH__
#define DATATYPE__CUH__

// #define USE_FLOAT


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

#endif