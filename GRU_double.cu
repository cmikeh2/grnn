#include "GRU_double.h"
#include "misc.h"

#include <cstring>
#include <iostream>
#include <cooperative_groups.h>
#include <math.h>
#include <cassert>
#include <limits>

using namespace cooperative_groups;

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-1.0f * x));
}

#define MM_BLOCK_SIZE 16
#define MM_REG_TILE 4
#define MM_TILE_SIZE 64

// This is a mostly optimized kernel for matrix multiplication
// The kernel uses a two tiered tiling mechanism that first tiles large
// tiles from global memory to shared memory. This shared memory tile is
// then used as the source to stream data into register arrays that perform
// a calculation on a 8x8 tile.

__global__ void matmul(float * A, float * B, float * C,
                       uint32_t M, uint32_t K, uint32_t N) {
  
  extern __shared__ float base[];
  float* bufferA = base;
  float* bufferB = &bufferA[MM_TILE_SIZE * MM_TILE_SIZE];

  float regA[MM_REG_TILE];
  float regB[MM_REG_TILE];
  float regC[MM_REG_TILE][MM_REG_TILE];
  
  uint32_t tidx = threadIdx.x;
  uint32_t tidy = threadIdx.y;
  uint32_t id = threadIdx.y * blockDim.x + threadIdx.x;
  uint32_t bidx = blockIdx.x;
  uint32_t bidy = blockIdx.y;

  // Number of rows that are traversed in a single fully coalesced load sequence
  constexpr uint32_t LOAD_STEPS = MM_TILE_SIZE * MM_TILE_SIZE / (MM_BLOCK_SIZE * MM_BLOCK_SIZE);
  constexpr uint32_t NUM_THREADS = MM_BLOCK_SIZE * MM_BLOCK_SIZE;
  
  // Zero the intermediate output
  for (uint32_t y = 0; y < MM_REG_TILE; y++) {
    for (uint32_t x = 0; x < MM_REG_TILE; x++) {
      regC[y][x] = 0.0f;
    }
  }

  for (uint32_t i = 0; i < K; i += MM_TILE_SIZE) {
    
    // Load lhs tile from global memory to shared memory (fully coalesced)
    #pragma unroll
    for (uint32_t j = 0; j < LOAD_STEPS; j++) {
      uint32_t index = j * NUM_THREADS + id;
      if (((bidy * MM_TILE_SIZE + index / MM_TILE_SIZE) < M) && ((i + index % MM_TILE_SIZE) < K)) {
        bufferA[index] = A[ (bidy * MM_TILE_SIZE + (index / MM_TILE_SIZE)) * K + i + index % MM_TILE_SIZE];
      } else {
        bufferA[index] = 0.0f;
      }
    }
    
    // Not necessary for correctness, but improves performance by avoiding thrashing shared memory
    __syncthreads();

    // Load rhs tile from global memory to shared memory (fully coalesced)
    #pragma unroll
    for (uint32_t j = 0; j < LOAD_STEPS; j++) {
      uint32_t index = j * NUM_THREADS + id;
      if (((i + index / MM_TILE_SIZE) < K) && ((bidx * MM_TILE_SIZE + index % MM_TILE_SIZE) < N)) {
        bufferB[index] = B[ ((index / MM_TILE_SIZE) + i) * N + bidx * MM_TILE_SIZE + index % MM_TILE_SIZE];
      } else {
        bufferB[index] = 0.0f;
      }
    }

    // Ensures all data is written from global memory to shared memory before it is streamed
    // into register arrays.
    __syncthreads();
    
    
      
    // Loop through full tile
    for (uint32_t j  = 0; j < MM_TILE_SIZE; j++) {
      
      // Load vector from lhs and rhs
      #pragma unroll
      for (uint32_t l = 0; l < MM_REG_TILE; l++) {
        regA[l] = bufferA[(tidy * MM_REG_TILE + l) * MM_TILE_SIZE + j];
        regB[l] = bufferB[j * MM_TILE_SIZE + tidx * MM_REG_TILE + l];
      }
      
      #pragma unroll
      // Perform a narrow matmul
      for (uint32_t y = 0; y < MM_REG_TILE; y++) {
        for (uint32_t x = 0; x < MM_REG_TILE; x++) {
          regC[y][x] += regA[y] * regB[x];
        }
      }
    }

    __syncthreads();
  }
 
  // Write register intermediates to shared memory (possibly unnecessary)
  for (uint32_t y = 0; y < MM_REG_TILE; y++) {
    for (uint32_t x = 0; x < MM_REG_TILE; x++) {
      bufferA[(tidy * MM_REG_TILE + y) * MM_TILE_SIZE + tidx * MM_REG_TILE + x] = regC[y][x];
    }
  }

  __syncthreads();

  
  for (uint32_t j = 0; j < LOAD_STEPS; j++) {
    uint32_t index = j * NUM_THREADS + id;
    if (((bidy * MM_TILE_SIZE + (index / MM_TILE_SIZE)) < M) && ((bidx * MM_TILE_SIZE + (index % MM_TILE_SIZE)) < N)) {
      C[ (bidy * MM_TILE_SIZE + (index / MM_TILE_SIZE)) * N + bidx * MM_TILE_SIZE +  (index % MM_TILE_SIZE)] = bufferA[index];
    }
  }

}


// This kernel assumes the input multiplications were precomputed in a large matrix-matrix multiplication
template<int HIDDEN_SIZE, int TILE_WIDTH, int TILE_HEIGHT, int NUM_GROUPS, int GROUP_THREADS, int BATCH_SIZE>
__global__ void gru_rnn(const float* precomputed_inputs,
                        const float* hidden_initializer,
                        const float* weights, 
                        const float* biases, 
                        float* r, 
                        float* output, 
                        volatile int* syncIn,
                        volatile int* syncOut,
                        uint32_t length) {
  
  // Indexing helpers
  int tid = threadIdx.x;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;
  int wg_id = tid / GROUP_THREADS;
  int r_id = tid / (2 * GROUP_THREADS);

  // LENGTH - How many weights for each output does a single thread need to store
  constexpr int LENGTH = (HIDDEN_SIZE + GROUP_THREADS - 1) / GROUP_THREADS;
  // BUFFER_SIZE - Number of elements to reserve in shared memory for each outout. Effectively
  // rounds up HIDDEN_SIZE to the next multiple of NUM_THREADS
  constexpr int BUFFER_SIZE = LENGTH * GROUP_THREADS;
  // OUTPUT_TILE_WIDTH - How many full elements are produced by the threadblock. At scheduling time,
  // must ensure that the launched configuration produces full elements within a single threadblock
  constexpr int OUTPUT_TILE_WIDTH = NUM_GROUPS * TILE_WIDTH / (GRU_GATES - 1);
  
  
  // Static shared memory allocation
  __shared__ float buffer_tile[TILE_HEIGHT][BUFFER_SIZE];
  __shared__ float z_gate[TILE_HEIGHT][OUTPUT_TILE_WIDTH];
  __shared__ float z_h_res[TILE_HEIGHT][OUTPUT_TILE_WIDTH];
  __shared__ float h_gate[2][TILE_HEIGHT][OUTPUT_TILE_WIDTH];
  
  // Weights in the register file
  float weights_reg[TILE_WIDTH][LENGTH];
  float h_weights_reg[TILE_WIDTH][LENGTH / 2];
  float outputs_reg[TILE_HEIGHT][TILE_WIDTH];
  float bias = 0.0f;
  float bias_h =  0.0f;
  float precompute = 0.0f;
  float precompute_h = 0.0f;
  const float * precomputed_offset;
  const float * precomputed_offset_h;

  // Cooperative group helpers
  thread_block bl = this_thread_block();
  thread_block_tile<GROUP_THREADS> work_group = tiled_partition<GROUP_THREADS>(bl);

  for (int i = 0; i < TILE_WIDTH; i++) {
    // Global gate id for fetching weights.
    // bidx * TILE_WIDTH * NUM_GROUPS -> the first gate index processed by the threadblock
    // wg_id * TILE_WIDTH -> the first gate index processed by a given warp within the threadblock
    // i -> current gate within the warp's assigned gates
    // These gate indexes will only refer to gates r and z, not the h gate
    int gate_id = bidx * TILE_WIDTH * NUM_GROUPS + wg_id * TILE_WIDTH + i;
    // The following lines transform the assigned r/z gate into its real index within the weight array.
    // Since we don't assign one of the gates, we undo the indexing to get a valid output_element.
    // We then determine which gate within the output the current assigned gate is.
    int output_element = (gate_id / (GRU_GATES - 1)) * GRU_GATES;
    int gate_index = gate_id % (GRU_GATES - 1);
    // Prevent segfaults
    if (output_element < HIDDEN_SIZE) {
      // 0 initialize rounded values. Better to have a single check now then on each recurrent iteration.
      for (int j = 0; j < LENGTH; j++) {
        if ( j * GROUP_THREADS + work_group.thread_rank() < HIDDEN_SIZE) {
          weights_reg[i][j] = weights[(output_element * GRU_GATES + gate_index) * HIDDEN_SIZE + j * GROUP_THREADS + work_group.thread_rank()];
        } else {
          weights_reg[i][j] = 0.f;
        }
      }
    }
  }
  
  for (int i = 0; i < TILE_WIDTH / 2; i++) {
    int output_element = bidx * NUM_GROUPS * TILE_WIDTH / 2 + r_id * TILE_WIDTH + i;
    int which_half = wg_id % 2;
    if (output_element < HIDDEN_SIZE) {
      for (int j = 0; j < LENGTH; j++) {
        if ( which_half * BUFFER_SIZE / 2 + j * GROUP_THREADS + work_group.thread_rank() < HIDDEN_SIZE) {
          h_weights_reg[i][j] = weights[output_element * GRU_GATES * HIDDEN_SIZE + 
                                        (GRU_GATES - 1) * HIDDEN_SIZE + 
                                        which_half * BUFFER_SIZE / 2 + 
                                        j * GROUP_THREADS + work_group.thread_rank()];
        } else {
          h_weights_reg[i][j] = 0.f;
        }
      }
    }
  }

  if (work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
    int x = work_group.thread_rank() % TILE_WIDTH;
    int y = work_group.thread_rank() / TILE_WIDTH;

    if ((bidx * TILE_WIDTH * NUM_GROUPS + wg_id * TILE_WIDTH + x) < HIDDEN_SIZE * (GRU_GATES - 1) && (bidy * TILE_HEIGHT + y < BATCH_SIZE)) {
      int output_element = ((bidx * TILE_WIDTH * NUM_GROUPS + wg_id * TILE_WIDTH + x) / (GRU_GATES - 1)) * GRU_GATES;
      int gate_index = (bidx * TILE_WIDTH * NUM_GROUPS + wg_id * TILE_WIDTH + x) % (GRU_GATES - 1);
      bias = biases[output_element * GRU_GATES + gate_index];

      precomputed_offset = precomputed_inputs;
      precomputed_offset += bidy * TILE_HEIGHT * HIDDEN_SIZE * GRU_GATES;
      precomputed_offset += y * HIDDEN_SIZE * GRU_GATES;
      precomputed_offset += output_element * GRU_GATES;
      precomputed_offset += gate_index;
      precompute = *precomputed_offset;
      precomputed_offset += BATCH_SIZE * HIDDEN_SIZE * GRU_GATES;
    }

  }

  if (tid < OUTPUT_TILE_WIDTH * TILE_HEIGHT) {
    int x = tid % OUTPUT_TILE_WIDTH;
    int y = tid / OUTPUT_TILE_WIDTH; 
    if ((bidx * OUTPUT_TILE_WIDTH + x < HIDDEN_SIZE) && (bidy * TILE_HEIGHT + y < BATCH_SIZE)) {
      bias_h = biases[(bidx * OUTPUT_TILE_WIDTH + x) * GRU_GATES + (GRU_GATES - 1)];

      precomputed_offset_h = precomputed_inputs;
      precomputed_offset_h += (GRU_GATES - 1);
      precomputed_offset_h += bidy * TILE_HEIGHT * HIDDEN_SIZE * GRU_GATES;
      precomputed_offset_h += y * HIDDEN_SIZE * GRU_GATES;
      precomputed_offset_h += (bidx * OUTPUT_TILE_WIDTH + x) * GRU_GATES;
      precompute_h = *precomputed_offset_h;
      precomputed_offset_h += BATCH_SIZE * HIDDEN_SIZE * GRU_GATES;
    }
  }
  
  #pragma unroll
  for (int j = 0; j < TILE_HEIGHT; j++) {
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH; i++) {
      outputs_reg[i][j] = 0.f;
    }
  }
  
  #pragma unroll
  for (int j = 0; j < TILE_HEIGHT; j++) {
    #pragma unroll
    for (int i = 0; i < BUFFER_SIZE; i += NUM_GROUPS * GROUP_THREADS) {
      if ( i + tid < HIDDEN_SIZE) {
        buffer_tile[j][i + tid] = hidden_initializer[(bidy * TILE_HEIGHT + j) * HIDDEN_SIZE + i + tid];
      } else if (i + tid < BUFFER_SIZE) {
        buffer_tile[j][i + tid] = 0.f;
      }
    }
  }

  for (int sequence_iteration = 0; sequence_iteration < length; sequence_iteration++) {
    
    #pragma unroll
    for (int k = 0; k < LENGTH; k++) {
      #pragma unroll
      for (int j = 0; j < TILE_HEIGHT; j++) {
        float val = buffer_tile[j][k * GROUP_THREADS + work_group.thread_rank()];
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
          outputs_reg[j][i] += val * weights_reg[i][k];
        }
      }
    }

    #pragma unroll
    for (int j = 0; j < TILE_HEIGHT; j++) {
      #pragma unroll
      for (int i = 0; i < TILE_WIDTH; i++) {
        #pragma unroll
        for (int k = 1; k < GROUP_THREADS; k *= 2) {
          outputs_reg[j][i] += work_group.shfl_xor(outputs_reg[j][i], k);
        }
      }
    }

    if (work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
      int reg_x = work_group.thread_rank() % TILE_WIDTH;
      int reg_y = work_group.thread_rank() / TILE_WIDTH;

      float val = outputs_reg[reg_y][reg_x] + bias + precompute;
      val = sigmoidf(val);

      int gate_id = (wg_id * TILE_WIDTH + reg_x) % (GRU_GATES - 1);
      int output_id = (bidx * NUM_GROUPS * TILE_WIDTH + wg_id * TILE_WIDTH + reg_x) / (GRU_GATES - 1);
      
      //r gate
      if (gate_id == 0) { 
        val = val * buffer_tile[reg_y][output_id];
        if (output_id < HIDDEN_SIZE) {
          r[(bidy * TILE_HEIGHT + reg_y) * HIDDEN_SIZE + output_id] = val;
        }
      } else {
        int smem_id = (wg_id * TILE_WIDTH + reg_x) / (GRU_GATES - 1);
        z_gate[reg_y][smem_id] = (1 - val);
        z_h_res[reg_y][smem_id] = val * buffer_tile[reg_y][output_id];
      }
    }

    // Synchronize between recurrent iterations
    if (tid == 0) {
      syncIn[bidy * gridDim.x + bidx] =  2 * sequence_iteration + 1;
    }

    __threadfence();
    
    #pragma unroll
    for (int j = 0; j < TILE_HEIGHT; j++) {
      #pragma unroll
      for (int i = 0; i < TILE_WIDTH; i++) {
        outputs_reg[i][j] = 0.f;
      }
    }

    if (bidx == 0) {
      if (tid < gridDim.x) {
        while ( syncIn[bidy * gridDim.x + tid] != 2 * sequence_iteration + 1) {
        }
      }

      __syncthreads();

      if (tid == 0) {
        syncOut[bidy] = 2 * sequence_iteration + 1;
      }
    } else {
      if (tid == 0) {
        while (syncOut[bidy] != 2 * sequence_iteration + 1) {
        }
      }
      __syncthreads();
    }
    
    #pragma unroll
    for (int j = 0; j < TILE_HEIGHT; j++) {
      #pragma unroll
      for (int i = 0; i < BUFFER_SIZE; i += NUM_GROUPS * GROUP_THREADS) {
        if ( i + tid < HIDDEN_SIZE ) {
          buffer_tile[j][i + tid] = r[bidy * TILE_HEIGHT * HIDDEN_SIZE + j * HIDDEN_SIZE + i + tid];
        } else if ( i + tid < BUFFER_SIZE) {
          buffer_tile[j][i + tid] = 0.f;
        }
      }
    }

    __syncthreads();
    
    int which_half = wg_id % 2;

    #pragma unroll
    for (int k = 0; k < LENGTH / 2; k++) {
      #pragma unroll
      for (int j = 0; j < TILE_HEIGHT; j++) {
        float val = buffer_tile[j][which_half * LENGTH * GROUP_THREADS / 2 + k * GROUP_THREADS + work_group.thread_rank()];
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
          outputs_reg[j][i] += val * h_weights_reg[i][k];
        }
      }
    }

    #pragma unroll
    for (int j = 0; j < TILE_HEIGHT; j++) {
      #pragma unroll
      for (int i = 0; i < TILE_WIDTH; i++) {
        #pragma unroll
        for (int k = 1; k < GROUP_THREADS; k *= 2) {
          outputs_reg[j][i] += work_group.shfl_xor(outputs_reg[j][i], k);
        }
      }
    }

    if (work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
      int x = work_group.thread_rank() % TILE_WIDTH;
      int y = work_group.thread_rank() / TILE_WIDTH;

      h_gate[which_half][y][r_id * TILE_WIDTH + x] = outputs_reg[y][x];
    }

    __syncthreads();
   
    if (tid < OUTPUT_TILE_WIDTH * TILE_HEIGHT) {
      int y = tid / OUTPUT_TILE_WIDTH;
      int smem_x = tid % OUTPUT_TILE_WIDTH;
      int global_x = bidx * OUTPUT_TILE_WIDTH + smem_x;
      if (global_x < HIDDEN_SIZE) {
        float val = tanh(h_gate[0][y][smem_x] + h_gate[1][y][smem_x] + precompute_h + bias_h);

        output[sequence_iteration * HIDDEN_SIZE * BATCH_SIZE + (bidy * TILE_HEIGHT + y) * HIDDEN_SIZE + global_x] = z_h_res[y][smem_x] + z_gate[y][smem_x] * val;
      }
    }
    
    if (sequence_iteration + 1 == length) break;
    
    if (tid == 0) {
      syncIn[bidy * gridDim.x + bidx] =  2 * sequence_iteration + 2;
    }

    if (work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
      precompute = *precomputed_offset;
      precomputed_offset += HIDDEN_SIZE * BATCH_SIZE * GRU_GATES;
    }

    if (tid < OUTPUT_TILE_WIDTH * TILE_HEIGHT) {
      precompute_h = *precomputed_offset_h;
      precomputed_offset_h += HIDDEN_SIZE * BATCH_SIZE * GRU_GATES;
    }

    __threadfence();

    if (bidx == 0) {
      if (tid < gridDim.x) {
        while ( syncIn[bidy * gridDim.x + tid] != 2 * sequence_iteration + 2) {
        }
      }

      __syncthreads();

      if (tid == 0) {
        syncOut[bidy] = 2 * sequence_iteration + 2;
      }
    } else {
      if (tid == 0) {
        while (syncOut[bidy] != 2 * sequence_iteration + 2) {
        }
      }
      __syncthreads();
    }
    
    #pragma unroll
    for (int j = 0; j < TILE_HEIGHT; j++) {
      #pragma unroll
      for (int i = 0; i < BUFFER_SIZE; i += NUM_GROUPS * GROUP_THREADS) {
        if ( i + tid < HIDDEN_SIZE ) {
          buffer_tile[j][i + tid] = output[sequence_iteration * HIDDEN_SIZE * BATCH_SIZE + bidy * TILE_HEIGHT * HIDDEN_SIZE + j * HIDDEN_SIZE + i + tid];
        } else if ( i + tid < BUFFER_SIZE) {
          buffer_tile[j][i + tid] = 0.f;
        }
      }
    }
  }
}

template<typename T>
void process_input_weights(T * output, std::vector<T*> weights, uint32_t input_size, uint32_t hidden_size) { 
  
  // Outside loop is the input size
  for (uint32_t j = 0; j < input_size; j++) {
    // Width of the input weight matrix
    for (uint32_t k = 0; k < hidden_size; k++) {
      // Colocate the weights for each element
      for (uint32_t i = 0; i < GRU_GATES; i++) {
        output[(j * hidden_size + k) * GRU_GATES + i] = weights.at(i)[j * hidden_size + k];
      }
    }
  }

}

template<typename T>
void process_hidden_weights(T * output, std::vector<T*> weights, uint32_t hidden_size) {
  
  // For each output element
  for (uint32_t j = 0; j < hidden_size; j++) {
    // For each gate
    for (uint32_t k = 0; k < GRU_GATES; k++) {
      // For each element for that gate
      for (uint32_t i = 0; i < hidden_size; i++) {
        output[j * GRU_GATES * hidden_size + k * hidden_size + i] = weights.at(3 + k)[i * hidden_size + j];
      }
    }
  }

}

template<typename T>
void process_biases(T * output, std::vector<T*> weights, uint32_t hidden_size) {

  // For each output element
  for (uint32_t k = 0; k < hidden_size; k++) {
    // Colocate the biases for each element
    for (uint32_t i = 0; i < GRU_GATES; i++) {
      output[k * GRU_GATES + i] = weights.at(i + 6)[k];
    }
  }

}

template <typename T>
void GRULayerDouble<T>::reset() {
  cudaFreeHost((void *) this->packed_input_weights);
  cudaFreeHost((void *) this->packed_hidden_weights);
  cudaFreeHost((void *) this->packed_biases);
  cudaFree((void *) this->packed_input_weights_gpu);
  cudaFree((void *) this->packed_hidden_weights_gpu);
  cudaFree((void *) this->packed_biases_gpu);
}


template<typename T>
uint32_t GRULayerDouble<T>::initialize() {
  
  uint32_t input_footprint = input_weight_footprint();
  uint32_t hidden_footprint = hidden_weight_footprint();
  uint32_t bias_footprint = bias_weight_footprint();

  cudaHostAlloc((void **) &(this->packed_input_weights), input_footprint, cudaHostAllocDefault); CUDA_ERR;
  cudaHostAlloc((void **) &(this->packed_hidden_weights), hidden_footprint, cudaHostAllocDefault); CUDA_ERR;
  cudaHostAlloc((void **) &(this->packed_biases), bias_footprint, cudaHostAllocDefault); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_input_weights_gpu), input_footprint); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_hidden_weights_gpu), hidden_footprint); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_biases_gpu), bias_footprint); CUDA_ERR;

  process_input_weights(this->packed_input_weights, this->host_weights, this->input_size, this->hidden_size);
  process_hidden_weights(this->packed_hidden_weights, this->host_weights, this->hidden_size);
  process_biases(this->packed_biases, this->host_weights, this->hidden_size);

  cudaMemcpy(this->packed_input_weights_gpu, this->packed_input_weights, input_footprint, cudaMemcpyHostToDevice); CUDA_ERR;
  cudaMemcpy(this->packed_hidden_weights_gpu, this->packed_hidden_weights, hidden_footprint, cudaMemcpyHostToDevice); CUDA_ERR;
  cudaMemcpy(this->packed_biases_gpu, this->packed_biases, bias_footprint, cudaMemcpyHostToDevice); CUDA_ERR;

  return 0;

}

template <typename T>
void GRUModelDouble<T>::reset() {

  for (auto& l: this->layers) {
    l.reset();
  }

  cudaFreeHost((void *) this->host_output);
  cudaFree((void *) this->gpu_output);
  
  cudaFree((void *) this->gpu_r);
  cudaFree((void *) this->gpu_inputs);
  cudaFree((void *) this->gpu_precompute);
  cudaFree((void *) this->gpu_syncIn);
  cudaFree((void *) this->gpu_syncOut);
}

template <typename T>
uint32_t GRUModelDouble<T>::initialize() {
  
  for (auto& l: this->layers) {
    uint32_t debug = l.initialize();
    if (debug != 0) {
      std::cout << "FAILURE\n";
      return debug;
    }
  }

  this->gpu_weights_input = this->layers[0].get_packed_input_weights_gpu();
  this->gpu_weights_hidden = this->layers[0].get_packed_hidden_weights_gpu();
  this->gpu_biases = this->layers[0].get_packed_biases_gpu();
  this->mm_k = this->initial_input_size;
  this->mm_n = this->output_size * GRU_GATES;

  // Output allocation, assume sequence length less than 200
  cudaHostAlloc((void **) &(this->host_output), this->output_size * this->batch_size * 200 * sizeof(T), cudaHostAllocDefault);
  cudaMalloc((void **) &(this->gpu_output), this->output_size * this->batch_size * 200 * sizeof(T));
  
  // Input allocations, assume sequence length less than 200
  cudaMalloc((void **) &(this->gpu_inputs), this->initial_input_size * this->batch_size * 200 * sizeof(T));
  cudaMalloc((void **) &(this->gpu_r), this->output_size * this->batch_size * sizeof(T));
  cudaMalloc((void **) &(this->gpu_precompute), this->output_size * this->batch_size * GRU_GATES * 200 * sizeof(T));

  // Hidden state initializer allocation
  cudaMalloc((void **) &(this->gpu_hidden_initializer), this->output_size * this->batch_size * sizeof(T));
  cudaMemset((void *)this->gpu_hidden_initializer, 0, this->output_size * this->batch_size * sizeof(T));

  // Synchronization buffer initialization
  cudaMalloc((void **) &(this->gpu_syncIn), 80 * sizeof(int));
  cudaMalloc((void **) &(this->gpu_syncOut), 80 * sizeof(int));
 
  //cudaFuncSetAttribute(gru_rnn, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SMEM); CUDA_ERR;
  cudaDeviceSetLimit(cudaLimitStackSize, 0); CUDA_ERR;

  this->paramsMM[0] = (void*) &(this->gpu_inputs);
  this->paramsMM[1] = (void*) &(this->gpu_weights_input);
  this->paramsMM[2] = (void*) &(this->gpu_precompute);
  this->paramsMM[4] = (void*) &(this->mm_k);
  this->paramsMM[5] = (void*) &(this->mm_n);

  this->paramsGRU[0] = (void*) &(this->gpu_precompute);
  this->paramsGRU[1] = (void*) &(this->gpu_hidden_initializer);
  this->paramsGRU[2] = (void*) &(this->gpu_weights_hidden);
  this->paramsGRU[3] = (void*) &(this->gpu_biases);
  this->paramsGRU[4] = (void*) &(this->gpu_r);
  this->paramsGRU[5] = (void*) &(this->gpu_output);
  this->paramsGRU[6] = (void*) &(this->gpu_syncIn);
  this->paramsGRU[7] = (void*) &(this->gpu_syncOut);

  return 0;

}

template <typename T>
void GRUModelDouble<T>::set_configuration(int x, int y, int g, int t) {
  this->tile_width = x;
  this->tile_height = y;
  this->num_groups = g;
  this->group_threads = t;
}



template <typename T>
float GRUModelDouble<T>::run_input(T* input, uint32_t * length) {
 
  this->mm_m = this->batch_size * *length;
  this->paramsMM[3] = (void *) &(this->mm_m);
  this->paramsGRU[8] = (void *) length;
 
  dim3 mm_grid = dim3((this->mm_n + MM_TILE_SIZE - 1) / MM_TILE_SIZE, (this->mm_m + MM_TILE_SIZE - 1) / MM_TILE_SIZE);
  dim3 mm_block = dim3(MM_BLOCK_SIZE, MM_BLOCK_SIZE);

  size_t mm_sm_requirement = MM_TILE_SIZE * MM_TILE_SIZE * 2 * sizeof(float);
  
  int effective_w = (this->num_groups * this->tile_width) / (GRU_GATES - 1);
  dim3 gru_rnn_grid = dim3((this->output_size + effective_w - 1) / effective_w, (this->batch_size + this->tile_height - 1) / this->tile_height);
  dim3 gru_rnn_block = dim3(this->num_groups * this->group_threads);
  
  unsigned block_size = gru_rnn_block.x;
  unsigned grid_size = gru_rnn_grid.x * gru_rnn_grid.y;
  
  void * kernel = (void *)gru_rnn<1024, 4, 5, 8, 32, 5>;

  int numBlocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, block_size, 0);

  if (grid_size > 80 * numBlocks) {
    printf("grid_size: %3d numBlocks: %3d block_size: %3d\n", grid_size, numBlocks * 80, block_size);
    return -std::numeric_limits<float>::infinity();
  }

  cudaEvent_t start, end;
  float elapsed;
  
  cudaMemcpy(this->gpu_inputs, input, this->initial_input_size * this->batch_size * *length * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemset(this->gpu_output, 0, this->output_size * this->batch_size * sizeof(T)); 
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  
  cudaLaunchKernel((void *)matmul, mm_grid, mm_block, this->paramsMM, mm_sm_requirement);
  cudaLaunchKernel(kernel, gru_rnn_grid, gru_rnn_block, this->paramsGRU);
  
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, start, end);
  
  cudaMemcpy(this->host_output, this->gpu_output, this->output_size * this->batch_size * sizeof(T), cudaMemcpyDeviceToHost);

#ifdef DEBUG
  for (int i = 0; i < this->batch_size; i++) {
    printf("Sequence %2d\n", i);
    for (int j = 0; j < this->output_size; j++) {
      printf("%f ", this->host_output[i * this->output_size + j]);
    }
    printf("\n");
  }
  printf("\n");
#endif

  cudaError_t err;
  cudaDeviceSynchronize();
  if ((err = cudaGetLastError()) != cudaSuccess) {
    printf("CUDA error: %d : %s : %s, line %d\n", err, cudaGetErrorString(err), __FILE__, __LINE__);
    return std::numeric_limits<float>::infinity();
  }
 
  return elapsed;
}

// Explicit template instantiations

template void process_input_weights<float>(float *, std::vector<float *>, uint32_t, uint32_t);
template void process_hidden_weights<float>(float *, std::vector<float *>, uint32_t);
template void process_biases<float>(float *, std::vector<float *>, uint32_t);
template uint32_t GRULayerDouble<float>::initialize();
template uint32_t GRUModelDouble<float>::initialize();
template void GRULayerDouble<float>::reset();
template void GRUModelDouble<float>::reset();
template void GRUModelDouble<float>::set_configuration(int, int, int, int);
template float GRUModelDouble<float>::run_input(float *, uint32_t *);
