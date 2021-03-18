#include "kernels.cuh"
#include "cub/cub.cuh"

#define MIN(x, y) ((x < y) ? x : y)

__device__ __forceinline__ elem_t siteLogLikelihood(nodeLikelihood root, elem_t* pi) {
    nodeLikelihood pi_root_sh;  //shift by exp(500)
    elem_t shift_max = -_MAX(_MAX(root.A, root.C), _MAX(root.G, root.T));

    pi_root_sh.A = pi[0] * exp(root.A + shift_max);
    pi_root_sh.C = pi[1] * exp(root.C + shift_max);
    pi_root_sh.G = pi[2] * exp(root.G + shift_max);
    pi_root_sh.T = pi[3] * exp(root.T + shift_max);

    return log(pi_root_sh.A + pi_root_sh.C + pi_root_sh.G + pi_root_sh.T) - shift_max;
}

__device__ __forceinline__ int encodeBase(char base) {
    if (base == 'A') return 0;
    else if (base == 'C') return 1;
    else if (base == 'G') return 2;
    else return 3;  // (base == 'T') 
}

__device__ __forceinline__ nodeLikelihood constructBaseLeaf(int encodedBase) {
    nodeLikelihood result;

    if (encodedBase == 0) result = {.A = 0, .C = -INFINITY, .G = -INFINITY, .T = -INFINITY};
    else if (encodedBase == 1) result = {.A = -INFINITY, .C = 0, .G = -INFINITY, .T = -INFINITY};
    else if (encodedBase == 2) result = {.A = -INFINITY, .C = -INFINITY, .G = 0, .T = -INFINITY};
    else if (encodedBase == 3) result = {.A = -INFINITY, .C = -INFINITY, .G = -INFINITY, .T = 0};
    return result;
}


// __global__ void build_expm(elem_t* expm_branch, elem_t* treeLengthArray, elem_t* d_rate_mat, const int totalNodeNum) {
//     // let's assume that expm(At) = I + At, approximation by taylor expansion
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= totalNodeNum)
//         return;
    
//     elem_t edge_length = treeLengthArray[idx];
//     for (int ii = 0; ii < 16; ii++) {
//         if (ii == 0 || ii == 5 || ii == 10 || ii == 15)  // diagonal
//             expm_branch[idx * 16 + ii] = 1 + (edge_length * d_rate_mat[ii]) ; 
//             //+ __powf(edge_length, 2) * d_rate_mat_square[ii] / 2 + __powf(edge_length, 3) * d_rate_mat_cubic[ii] / 6 ;
//         else
//             expm_branch[idx * 16 + ii] = (edge_length * d_rate_mat[ii]) ; 
//             //+ __powf(edge_length, 2) * d_rate_mat_square[ii] / 2 + __powf(edge_length, 3) * d_rate_mat_cubic[ii] / 6 ;
//     }
//     return;
// }



__global__ void nodeValInit(nodeLikelihood* nodeVal, char* __restrict__ seq, int* node_level, int totalNodeNum, 
                            int seqLength, int seqNum) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNodeNum)
        return;
    const int seqcolIdx = blockIdx.y;
    if (seqcolIdx >= seqLength)
        return;

    // cooperative_groups::grid_group g = cooperative_groups::this_grid();
    if(idx < seqNum) {  // initialize childs
        nodeVal[seqcolIdx * totalNodeNum + idx] = constructBaseLeaf(encodeBase(seq[seqcolIdx * seqNum + idx]));
    }
    else {
        nodeVal[seqcolIdx * totalNodeNum + idx] = {.A = 0, .C = 0, .G = 0, .T = 0};  // multiply-cumulated
    }
}


__global__ void computePerSiteScore(nodeLikelihood* nodeVal, elem_t* treeLengthArray, 
                                    elem_t* d_rate_mat, elem_t* d_rate_mat_square, elem_t* d_rate_mat_cubic, 
                                    int* __restrict__ treeArray, int* __restrict__ node_level,
                                    int totalNodeNum, int seqLength, int seqNum, int curr_depth) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNodeNum)
        return;
    // const int seqcolIdx = blockIdx.y;
    // if (seqcolIdx >= seqLength)
    //     return;
        
    __shared__ elem_t rate_mat_cache[16];
    __shared__ elem_t rate_mat_square_cache[16];
    __shared__ elem_t rate_mat_cubic_cache[16];
    const int tid = threadIdx.x;
    if (tid < 16) {
        rate_mat_cache[tid] = d_rate_mat[tid];
        rate_mat_square_cache[tid] = d_rate_mat_square[tid];
        rate_mat_cubic_cache[tid] = d_rate_mat_cubic[tid];
    }
    __syncthreads();

    elem_t edge_length = treeLengthArray[idx];
    elem_t expm_p[16];
    for (int ii = 0 ; ii < 16; ii ++) {
        if (ii == 0 || ii == 5 || ii == 10 || ii == 15)  // diagonal
            expm_p[ii] = 1 + (edge_length * rate_mat_cache[ii])
            + pow(edge_length, 2) * rate_mat_square_cache[ii] / 2 + pow(edge_length, 3) * rate_mat_cubic_cache[ii] / 6 ;
        else
            expm_p[ii] = (edge_length * rate_mat_cache[ii])
            + pow(edge_length, 2) * rate_mat_square_cache[ii] / 2 + pow(edge_length, 3) * rate_mat_cubic_cache[ii] / 6 ;
    }

    for (int seqcolIdx = blockIdx.y; seqcolIdx < seqLength; seqcolIdx += blockDim.y * gridDim.y) 
    {
        if (node_level[idx] == curr_depth) {
            int parent_ = treeArray[idx];
            nodeLikelihood child = nodeVal[seqcolIdx * totalNodeNum + idx];
            nodeLikelihood shft_child;

            elem_t shift_max = -_MAX(_MAX(child.A, child.C), _MAX(child.G, child.T));
            // elem_t* expm_p = expm_branch + idx * 16;

            shft_child.A = exp(child.A + shift_max);
            shft_child.C = exp(child.C + shift_max);
            shft_child.G = exp(child.G + shift_max);
            shft_child.T = exp(child.T + shift_max);
            
            atomicAdd(&(nodeVal[seqcolIdx * totalNodeNum + parent_].A), log(expm_p[0] * shft_child.A + expm_p[4] * shft_child.C + expm_p[8]  * shft_child.G + expm_p[12] * shft_child.T)-shift_max);
            atomicAdd(&(nodeVal[seqcolIdx * totalNodeNum + parent_].C), log(expm_p[1] * shft_child.A + expm_p[5] * shft_child.C + expm_p[9]  * shft_child.G + expm_p[13] * shft_child.T)-shift_max);
            atomicAdd(&(nodeVal[seqcolIdx * totalNodeNum + parent_].G), log(expm_p[2] * shft_child.A + expm_p[6] * shft_child.C + expm_p[10] * shft_child.G + expm_p[14] * shft_child.T)-shift_max);
            atomicAdd(&(nodeVal[seqcolIdx * totalNodeNum + parent_].T), log(expm_p[3] * shft_child.A + expm_p[7] * shft_child.C + expm_p[11] * shft_child.G + expm_p[15] * shft_child.T)-shift_max);
            // if (seqcolIdx == 0) {
            //     for (int i = 0 ; i < 50; ++i) {
            //         printf("%f %f %f %f \n", nodeVal[seqcolIdx * totalNodeNum + parent_].A, nodeVal[seqcolIdx * totalNodeNum + parent_].C, nodeVal[seqcolIdx * totalNodeNum + parent_].G, nodeVal[seqcolIdx * totalNodeNum + parent_].T);
            //     } printf("\n");
            // }
        }
    }
}

__global__ void rootScoreCalc(float* treeSiteScore, int* node_level, elem_t* pi, nodeLikelihood* nodeVal, int totalNodeNum, const int seqLength) {
    // const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx >= seqLength)
    //     return;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNodeNum)
        return;
    // const int seqcolIdx = blockIdx.y;
    // if (seqcolIdx >= seqLength)
    //     return;
        
    __shared__ elem_t pi_cache[4];
    const int tid = threadIdx.x;
    if (tid < 4) {
        pi_cache[tid] = pi[tid];
    }
    __syncthreads();

    for (int seqcolIdx = blockIdx.y; seqcolIdx < seqLength; seqcolIdx += blockDim.y * gridDim.y) 
    {
        if (node_level[idx] == 0)
            treeSiteScore[seqcolIdx] = siteLogLikelihood(nodeVal[seqcolIdx * totalNodeNum + idx], pi_cache);
    }
}

void GPUInitialize(Params &params, char* transpose, int seq_length, int seq_num) {

    // GPU Allocation
    char* d_seqs = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_seqs, seq_length * seq_num * sizeof(char)));
    elem_t* d_odata = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_odata, sizeof(elem_t)));
    // elem_t* d_expm_branch = NULL;
    // HANDLE_ERROR(cudaMalloc((void **)&d_expm_branch, 16 * tree_total_node_num * sizeof(elem_t)));
    elem_t* d_treeSiteScore = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_treeSiteScore, seq_length * sizeof(elem_t)));
    // nodeLikelihood* d_nodeVal = NULL;
    // HANDLE_ERROR(cudaMalloc((void **)&d_nodeVal, seq_length * tree_total_node_num * sizeof(nodeLikelihood)));
    elem_t* d_pi = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_pi, 4 * sizeof(elem_t)));
    elem_t* d_rate_mat = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_rate_mat, 16 * sizeof(elem_t)));
    elem_t* d_rate_mat_square = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_rate_mat_square, 16 * sizeof(elem_t)));
    elem_t* d_rate_mat_cubic = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_rate_mat_cubic, 16 * sizeof(elem_t)));
    // int* d_treeArray = NULL;
    // HANDLE_ERROR(cudaMalloc((void **)&d_treeArray, tree_total_node_num * sizeof(int)));
    // int* d_node_level = NULL;
    // HANDLE_ERROR(cudaMalloc((void **)&d_node_level, tree_total_node_num * sizeof(int)));
    // elem_t* d_treeLengthArray = NULL;
    // HANDLE_ERROR(cudaMalloc((void **)&d_treeLengthArray, tree_total_node_num * sizeof(elem_t)));

    // Pass pointer to the Params
    params.d_seqs = d_seqs;
    params.d_odata = d_odata;
    // params.d_expm_branch = d_expm_branch;
    params.d_treeSiteScore = d_treeSiteScore;
    // params.d_nodeVal = d_nodeVal;
    params.d_pi = d_pi;
    params.d_rate_mat = d_rate_mat;
    params.d_rate_mat_square = d_rate_mat_square;
    params.d_rate_mat_cubic = d_rate_mat_cubic;
    // params.d_treeArray = d_treeArray;
    // params.d_node_level = d_node_level;
    // params.d_treeLengthArray = d_treeLengthArray;

    // Copy sequences
    HANDLE_ERROR(cudaMemcpy(d_seqs, transpose, seq_length * seq_num * sizeof(char), cudaMemcpyHostToDevice));

    return;
}


void GPUDestroy(Params &params) {
    HANDLE_ERROR(cudaFree(params.d_seqs));
    HANDLE_ERROR(cudaFree(params.d_odata));
    // HANDLE_ERROR(cudaFree(params.d_expm_branch));
    HANDLE_ERROR(cudaFree(params.d_treeSiteScore));
    // HANDLE_ERROR(cudaFree(params.d_nodeVal));
    HANDLE_ERROR(cudaFree(params.d_pi));
    HANDLE_ERROR(cudaFree(params.d_rate_mat));
    HANDLE_ERROR(cudaFree(params.d_rate_mat_square));
    HANDLE_ERROR(cudaFree(params.d_rate_mat_cubic));
    // HANDLE_ERROR(cudaFree(params.d_treeArray));
    // HANDLE_ERROR(cudaFree(params.d_node_level));
    // HANDLE_ERROR(cudaFree(params.d_treeLengthArray));
}

void cuda_maxll_score(elem_t& output_score, Params &params, int* treeArray, elem_t* treeLengthArray, int* node_level, elem_t* rate_mat, 
                        elem_t* pi, int tree_total_node_num, int seq_length, int seq_num) {

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    int h_max_depth = *std::max_element(node_level, node_level + tree_total_node_num);

    elem_t rate_mat_square[16] = {0, };
    elem_t rate_mat_cubic[16] = {0, };
    for(int i=0; i<4; ++i)
        for(int j=0; j<4; ++j)
            for(int k=0; k<4; ++k) {
                rate_mat_square[i * 4 + j] += rate_mat[i * 4 + k] * rate_mat[k * 4 + j];
            }
    for(int i=0; i<4; ++i)
        for(int j=0; j<4; ++j)
            for(int k=0; k<4; ++k) {
                rate_mat_cubic[i * 4 + j] += rate_mat_square[i * 4 + k] * rate_mat[k * 4 + j];
            }


    char* d_seqs = params.d_seqs;
    elem_t* d_odata = params.d_odata;
    // elem_t* d_expm_branch = params.d_expm_branch;
    elem_t* d_treeSiteScore = params.d_treeSiteScore;
    // nodeLikelihood* d_nodeVal = params.d_nodeVal;
    elem_t* d_pi = params.d_pi;
    elem_t* d_rate_mat = params.d_rate_mat;
    elem_t* d_rate_mat_square = params.d_rate_mat_square;
    elem_t* d_rate_mat_cubic = params.d_rate_mat_cubic;
    // int* d_treeArray = params.d_treeArray;
    // int* d_node_level = params.d_node_level;
    // elem_t* d_treeLengthArray = params.d_treeLengthArray;

    // elem_t* d_expm_branch = NULL;
    // HANDLE_ERROR(cudaMalloc((void **)&d_expm_branch, 16 * tree_total_node_num * sizeof(elem_t)));
    nodeLikelihood* d_nodeVal = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_nodeVal, seq_length * tree_total_node_num * sizeof(nodeLikelihood)));
    int* d_treeArray = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_treeArray, tree_total_node_num * sizeof(int)));
    int* d_node_level = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_node_level, tree_total_node_num * sizeof(int)));
    elem_t* d_treeLengthArray = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_treeLengthArray, tree_total_node_num * sizeof(elem_t)));

    HANDLE_ERROR(cudaMemcpy(d_pi, pi, 4 * sizeof(elem_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_rate_mat, rate_mat, 16 * sizeof(elem_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_rate_mat_square, rate_mat_square, 16 * sizeof(elem_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_rate_mat_cubic, rate_mat_cubic, 16 * sizeof(elem_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_treeArray, treeArray, tree_total_node_num * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_node_level, node_level, tree_total_node_num * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_treeLengthArray, treeLengthArray, tree_total_node_num * sizeof(elem_t), cudaMemcpyHostToDevice));
    
    /// Kernel Launch ///
    dim3 computeMLSiteBlocks((tree_total_node_num + N_THREADS - 1) / N_THREADS, MIN(seq_length, prop.maxGridSize[1]));
    dim3 computeMLSiteTPB(N_THREADS, 1, 1);

    // build_expm<<<(tree_total_node_num+N_THREADS-1)/N_THREADS , N_THREADS>>>(d_expm_branch, d_treeLengthArray, d_rate_mat, tree_total_node_num);
    nodeValInit<<<computeMLSiteBlocks, computeMLSiteTPB>>>(d_nodeVal, d_seqs, d_node_level, tree_total_node_num, seq_length, seq_num);
    for (int curr_depth = h_max_depth; curr_depth > 0; curr_depth--) {
        computePerSiteScore<<<computeMLSiteBlocks, computeMLSiteTPB>>>(d_nodeVal, d_treeLengthArray, d_rate_mat, d_rate_mat_square, d_rate_mat_cubic, 
                                                                    d_treeArray, d_node_level,
                                                                    tree_total_node_num, seq_length, seq_num, curr_depth);
    }
    rootScoreCalc<<<computeMLSiteBlocks, computeMLSiteTPB>>>(d_treeSiteScore, d_node_level, d_pi, d_nodeVal, tree_total_node_num, seq_length);
    // rootScoreCalc<<<(seq_length + N_THREADS - 1)/N_THREADS, N_THREADS>>>(d_treeSiteScore, d_pi, d_nodeVal, tree_total_node_num, seq_length);

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_treeSiteScore, d_odata, seq_length);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_treeSiteScore, d_odata, seq_length);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaFree(d_temp_storage));
    
    cudaMemcpy(&output_score, d_odata, sizeof(elem_t), cudaMemcpyDeviceToHost);

    // HANDLE_ERROR(cudaFree(params.d_seqs));
    // HANDLE_ERROR(cudaFree(params.d_odata));
    // HANDLE_ERROR(cudaFree(d_expm_branch));
    // HANDLE_ERROR(cudaFree(params.d_treeSiteScore));
    HANDLE_ERROR(cudaFree(d_nodeVal));
    // HANDLE_ERROR(cudaFree(params.d_pi));
    // HANDLE_ERROR(cudaFree(params.d_rate_mat));
    HANDLE_ERROR(cudaFree(d_treeArray));
    HANDLE_ERROR(cudaFree(d_node_level));
    HANDLE_ERROR(cudaFree(d_treeLengthArray));

    return;
}
