#include "kernels.cuh"
#include "cub/cub.cuh"

__device__ __forceinline__ elem_t siteLogLikelihood(nodeLikelihood root, elem_t* pi) {
    nodeLikelihood pi_root_sh;  //shift by exp(500)
    elem_t shift_max = -_MAX(_MAX(root.A, root.C), _MAX(root.G, root.T));

    pi_root_sh.A = pi[0] * __expf(root.A + shift_max);
    pi_root_sh.C = pi[1] * __expf(root.C + shift_max);
    pi_root_sh.G = pi[2] * __expf(root.G + shift_max);
    pi_root_sh.T = pi[3] * __expf(root.T + shift_max);

    return __logf(pi_root_sh.A + pi_root_sh.C + pi_root_sh.G + pi_root_sh.T) - shift_max;
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



__global__ void build_expm(elem_t* expm_branch, elem_t* treeLengthArray, elem_t* d_rate_mat, const int totalNodeNum) {
    // let's assume that expm(At) = I + At, approximation by taylor expansion
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNodeNum)
        return;
    
    elem_t edge_length = treeLengthArray[idx];
    for (int ii = 0; ii < 16; ii++) {
        if (ii == 0 || ii == 5 || ii == 10 || ii == 15)  // diagonal
            expm_branch[idx * 16 + ii] = 1 + (edge_length * d_rate_mat[ii]) ; 
            //+ __powf(edge_length, 2) * d_rate_mat_square[ii] / 2 + __powf(edge_length, 3) * d_rate_mat_cubic[ii] / 6 ;
        else
            expm_branch[idx * 16 + ii] = (edge_length * d_rate_mat[ii]) ; 
            //+ __powf(edge_length, 2) * d_rate_mat_square[ii] / 2 + __powf(edge_length, 3) * d_rate_mat_cubic[ii] / 6 ;
    }
    return;
}


__global__ void nodeValInit(nodeLikelihood* nodeVal, char* seq, int* __restrict__ node_level, int totalNodeNum, 
                            int seqLength, int seqNum, int max_node_level) {
    // const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx >= totalNodeNum)
    //     return;
    // const int seqcolIdx = blockIdx.y;
    // if (seqcolIdx >= seqLength)
    //     return;

    // // cooperative_groups::grid_group g = cooperative_groups::this_grid();
    // int curr_node_level = node_level[idx];
    // if(curr_node_level == max_node_level) {  // initialize childs
    //     nodeVal[seqcolIdx * totalNodeNum + idx] = constructBaseLeaf(encodeBase(seq[seqcolIdx * seqNum + ]));
    // }
    // else {
    //     nodeVal[seqcolIdx * totalNodeNum + idx] = {.A = 0, .C = 0, .G = 0, .T = 0};  // multiply-cumulated
    // }
    const int seqcolIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seqcolIdx >= seqLength)
        return;
    
    int j = 0;
    for (int idx = 0 ; idx < totalNodeNum; idx++) {
        if(node_level[idx] == max_node_level) {  // initialize childs
            nodeVal[seqcolIdx * totalNodeNum + idx] = constructBaseLeaf(encodeBase(seq[seqcolIdx * seqNum + j]));
            j++;
        }
        else {
            nodeVal[seqcolIdx * totalNodeNum + idx] = {.A = 0, .C = 0, .G = 0, .T = 0};  // multiply-cumulated
        }
    }
}


__global__ void computePerSiteScore(nodeLikelihood* nodeVal, elem_t* __restrict__ expm_branch, 
                                    int* __restrict__ treeArray, int* __restrict__ node_level,
                                    int totalNodeNum, int seqLength, int seqNum, int curr_depth) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNodeNum)
        return;
    const int seqcolIdx = blockIdx.y;
    if (seqcolIdx >= seqLength)
        return;

    if (node_level[idx] == curr_depth) {
        int parent_ = treeArray[idx];
        nodeLikelihood child = nodeVal[seqcolIdx * totalNodeNum + idx];
        nodeLikelihood shft_child;

        elem_t shift_max = -_MAX(_MAX(child.A, child.C), _MAX(child.G, child.T));
        elem_t* expm_p = expm_branch + idx * 16;

        shft_child.A = __expf(child.A + shift_max);
        shft_child.C = __expf(child.C + shift_max);
        shft_child.G = __expf(child.G + shift_max);
        shft_child.T = __expf(child.T + shift_max);
        
        atomicAdd(&(nodeVal[seqcolIdx * totalNodeNum + parent_].A), __logf(expm_p[0] * shft_child.A + expm_p[4] * shft_child.C + expm_p[8]  * shft_child.G + expm_p[12] * shft_child.T)-shift_max);
        atomicAdd(&(nodeVal[seqcolIdx * totalNodeNum + parent_].C), __logf(expm_p[1] * shft_child.A + expm_p[5] * shft_child.C + expm_p[9]  * shft_child.G + expm_p[13] * shft_child.T)-shift_max);
        atomicAdd(&(nodeVal[seqcolIdx * totalNodeNum + parent_].G), __logf(expm_p[2] * shft_child.A + expm_p[6] * shft_child.C + expm_p[10] * shft_child.G + expm_p[14] * shft_child.T)-shift_max);
        atomicAdd(&(nodeVal[seqcolIdx * totalNodeNum + parent_].T), __logf(expm_p[3] * shft_child.A + expm_p[7] * shft_child.C + expm_p[11] * shft_child.G + expm_p[15] * shft_child.T)-shift_max);
    }
}

__global__ void rootScoreCalc(float* treeSiteScore, elem_t* pi, nodeLikelihood* nodeVal, int totalNodeNum, const int seqLength) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seqLength)
        return;
    treeSiteScore[idx] = siteLogLikelihood(nodeVal[idx * totalNodeNum + 0], pi);
}


elem_t cuda_maxll_score(char* seqs, int* treeArray, elem_t* treeLengthArray, int* node_level, elem_t* rate_mat, 
                        elem_t* pi, int tree_total_node_num, int seq_length, int seq_num) {

    int h_max_depth = *std::max_element(node_level, node_level + tree_total_node_num);
    
    char* d_seqs = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_seqs, seq_length * seq_num * sizeof(char)));
    elem_t* d_odata = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_odata, sizeof(elem_t)));
    elem_t* d_expm_branch = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_expm_branch, 16 * tree_total_node_num * sizeof(elem_t)));
    elem_t* d_treeSiteScore = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_treeSiteScore, seq_length * sizeof(elem_t)));
    nodeLikelihood* d_nodeVal = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_nodeVal, seq_length * tree_total_node_num * sizeof(nodeLikelihood)));
    elem_t* d_pi = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_pi, 4 * sizeof(elem_t)));
    elem_t* d_rate_mat = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_rate_mat, 16 * sizeof(elem_t)));
    int* d_treeArray = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_treeArray, tree_total_node_num * sizeof(int)));
    int* d_node_level = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_node_level, tree_total_node_num * sizeof(int)));
    elem_t* d_treeLengthArray = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_treeLengthArray, tree_total_node_num * sizeof(elem_t)));

    HANDLE_ERROR(cudaMemcpy(d_pi, pi, 4 * sizeof(elem_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_rate_mat, rate_mat, 16 * sizeof(elem_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_treeArray, treeArray, tree_total_node_num * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_node_level, node_level, tree_total_node_num * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_seqs, seqs, seq_length * seq_num * sizeof(char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_treeLengthArray, treeLengthArray, tree_total_node_num * sizeof(elem_t), cudaMemcpyHostToDevice));
    
    /// Kernel Launch ///
    dim3 computeMLSiteBlocks((tree_total_node_num + N_THREADS - 1) / N_THREADS, seq_length);
    dim3 computeMLSiteTPB(N_THREADS, 1, 1);

    build_expm<<<(tree_total_node_num+N_THREADS-1)/N_THREADS , N_THREADS>>>(d_expm_branch, d_treeLengthArray, d_rate_mat, tree_total_node_num);
    nodeValInit<<<(seq_length + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_nodeVal, d_seqs, d_node_level, tree_total_node_num, seq_length, seq_num, h_max_depth);
    for (int curr_depth = h_max_depth; curr_depth > 0; curr_depth--) {
        computePerSiteScore<<<computeMLSiteBlocks, computeMLSiteTPB>>>(d_nodeVal, d_expm_branch, d_treeArray, d_node_level,
                                                                    tree_total_node_num, seq_length, seq_num, curr_depth);
    }
    rootScoreCalc<<<(seq_length + N_THREADS - 1)/N_THREADS, N_THREADS>>>(d_treeSiteScore, d_pi, d_nodeVal, tree_total_node_num, seq_length);

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_treeSiteScore, d_odata, seq_length);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_treeSiteScore, d_odata, seq_length);

    elem_t* h_odata = (elem_t *)malloc(sizeof(elem_t));
    HANDLE_ERROR(cudaMemcpy(h_odata, d_odata, sizeof(elem_t), cudaMemcpyDeviceToHost));
    
    cudaFree(d_pi);
    cudaFree(d_rate_mat);
    cudaFree(d_treeArray);
    cudaFree(d_treeLengthArray);
    cudaFree(d_node_level);
    cudaFree(d_expm_branch);
    cudaFree(d_odata);
    cudaFree(d_seqs);
    cudaFree(d_treeSiteScore);
    cudaFree(d_nodeVal);

    free(h_odata);

    return h_odata[0];
}
