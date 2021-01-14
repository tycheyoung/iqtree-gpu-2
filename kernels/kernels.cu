#include "kernels.cuh"

__global__ void testtest(int i ) {
    int idx = threadIdx.x;
    printf("%d, %d\n", idx, i);
}

void testwrapper() {
    testtest<<<1, 1>>>(2);
    printf("Hello!\n");
    return;
}
