#include "GRU_single.h"
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

template<int HIDDEN_SIZE, int TILE_WIDTH, int TILE_HEIGHT, int NUM_GROUPS, int GROUP_THREADS, int BATCH_SIZE>
__global__ void gru_rnn(const float* precomputed_inputs,
                        const float* hidden_initializer,
                        const float* weights_r,
                        const float* weights_zh,
                        const float* biases_r,
                        const float* biases_zh,
                        float* r_buf,
                        float* output,
                        volatile int* syncIn,
                        volatile int* syncOut,
                        uint32_t length) {
  // Indexing Helpers
  int tid = threadIdx.x;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;
  // Number of weights stored per tile width
  constexpr int LENGTH = (HIDDEN_SIZE + GROUP_THREADS - 1) / GROUP_THREADS;
  // Number of elements to reserve in shared memory for each output. Effectively 
  // rounds up HIDDEN_SIZE to multiple of GROUP_THREADS
  constexpr int BUFFER_SIZE = LENGTH * GROUP_THREADS;
  // Number of elements horizontally produced by a single threadblock
  constexpr int OUTPUT_TILE_WIDTH = TILE_WIDTH * NUM_GROUPS / (GRU_GATES - 1);
  // Number of threads in a launched block
  constexpr int NUM_THREADS = NUM_GROUPS * GROUP_THREADS;
  // Number of outputs per tile row a single thread must compute for the partial sums of r values
  constexpr int ELEMS_PER_THREAD = (HIDDEN_SIZE + NUM_THREADS - 1) / NUM_THREADS;
  // Number of partial sums produced by the kernel for each input in the batch for the r gate
  constexpr int NUM_PARTIALS = (HIDDEN_SIZE + OUTPUT_TILE_WIDTH - 1) / OUTPUT_TILE_WIDTH;
  
  // Determines whether a group is the h gate or the z gate
  int g_type = 2 * tid / (NUM_THREADS);
  int wg_id = (tid % (NUM_THREADS / 2)) / GROUP_THREADS;

  // Shared memory workspaces
  __shared__ float h_tile[TILE_HEIGHT][BUFFER_SIZE];
  __shared__ float r_tile[TILE_HEIGHT][BUFFER_SIZE];
  __shared__ float z_gate[TILE_HEIGHT][OUTPUT_TILE_WIDTH];
  __shared__ float h_gate[TILE_HEIGHT][OUTPUT_TILE_WIDTH];
  
  // Tiled weights for z or h gates
  float weights_reg[TILE_WIDTH][LENGTH];
  // Weight for the r gate
  float weights_reg_r[OUTPUT_TILE_WIDTH][ELEMS_PER_THREAD];
  float outputs_reg[TILE_HEIGHT][TILE_WIDTH];
  float bias = 0.f;
  float bias_r[ELEMS_PER_THREAD];
  float precompute = 0.f;
  const float* precomputed_offset = precomputed_inputs;
  const float* precomputed_offset_r = precomputed_inputs + bidy * TILE_HEIGHT * HIDDEN_SIZE * GRU_GATES;
  
  // Work group declaration
  thread_block bl = this_thread_block();
  thread_block_tile<GROUP_THREADS> work_group = tiled_partition<GROUP_THREADS>(bl);
  
  // Stream appropriate weights for element_id and gate_id into the register file
  for (int i = 0; i < TILE_WIDTH; i++) {
    int group_id = bidx * OUTPUT_TILE_WIDTH + wg_id * TILE_WIDTH + i;

    if (group_id < HIDDEN_SIZE){
      for (int j = 0; j < LENGTH; j++) {
        if ( j * GROUP_THREADS + work_group.thread_rank() < HIDDEN_SIZE) {
          weights_reg[i][j] = weights_zh[(group_id * (GRU_GATES - 1) + g_type) * HIDDEN_SIZE + j * GROUP_THREADS + work_group.thread_rank()];
        } else {
          weights_reg[i][j] = 0.f;
        }
      }
    }
  }
  
  // Load biases and define time independent offsets
  if (work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
    int group_id = (bidx * OUTPUT_TILE_WIDTH + wg_id * TILE_WIDTH + work_group.thread_rank()) % TILE_WIDTH;
    int gate_id = (bidx * TILE_WIDTH * NUM_GROUPS + wg_id * TILE_WIDTH + work_group.thread_rank()) % TILE_WIDTH;
    if (group_id < HIDDEN_SIZE) {
      bias = biases_zh[group_id * (GRU_GATES - 1) + g_type];

      precomputed_offset += bidy * TILE_HEIGHT * HIDDEN_SIZE * GRU_GATES;
      precomputed_offset += group_id * GRU_GATES; 
      precomputed_offset += g_type + 1;
      precomputed_offset += (work_group.thread_rank() / TILE_WIDTH) * HIDDEN_SIZE;
    } else {
      bias = 0.f;
    }
  }

  // Stream weights for the r gate into the register file
  for (int j = 0; j < OUTPUT_TILE_WIDTH; j++) {
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
      if ( i * NUM_THREADS + tid < HIDDEN_SIZE) {
        weights_reg_r[j][i] = weights_r[bidx * OUTPUT_TILE_WIDTH * HIDDEN_SIZE + j * HIDDEN_SIZE + i * NUM_THREADS + tid];
      } else {
        weights_reg_r[j][i] = 0.f;
      }
    }
  }
  
  // Stream biases for the r_gate into the register file
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
    if ( i * NUM_THREADS + tid < HIDDEN_SIZE) {
      bias_r[i] = biases_r[i * NUM_THREADS + tid];
    } else {
      bias_r[i] = 0.f;
    }
  }
  
  // For the first iteration, load initial hidden state into the hidden tile.
  // This doesn't need to be repeated because once the recurrent pattern is established
  // the loop will populate the hidden_tile as the necessary outputs are produced.
  if ( tid < OUTPUT_TILE_WIDTH * TILE_HEIGHT ) {
    int x = tid % OUTPUT_TILE_WIDTH;
    int y = tid / OUTPUT_TILE_WIDTH;
    h_tile[y][x] = hidden_initializer[(bidy * TILE_HEIGHT + y) * HIDDEN_SIZE + bidx * OUTPUT_TILE_WIDTH + x];
  }

  // Hidden state initialization
  #pragma unroll
  for (int j = 0; j < TILE_HEIGHT; j++) {
    #pragma unroll
    for (int i = 0; i < BUFFER_SIZE; i += NUM_GROUPS * GROUP_THREADS) {
      if ( i + tid < HIDDEN_SIZE) {
        h_tile[j][i + tid] = hidden_initializer[(bidy * TILE_HEIGHT + j) * HIDDEN_SIZE + i + tid];
      } else if (i + tid < BUFFER_SIZE) {
        h_tile[j][i + tid] = 0.f;
      }
    }
  }

  __syncthreads();
  
  // Main recurrent loop
  for (int sequence_iteration = 0; sequence_iteration < length; sequence_iteration++) {

    // Produce partial dot products for the r gate
    for (int k = 0; k < TILE_HEIGHT; k++) {
      float r_dot_products[ELEMS_PER_THREAD];
      
      //  Zero initialize partials
      for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        r_dot_products[i] = 0.f;
      }
      
      // Process hidden_tile elements, getting maximum reuse
      for (int j = 0; j < OUTPUT_TILE_WIDTH; j++) {
        float rhs = h_tile[k][bidx * OUTPUT_TILE_WIDTH + j];
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
          r_dot_products[i] += weights_reg_r[j][i] * rhs;
        }
      }
      
      // Write to the global buffer
      for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        if ( i * NUM_THREADS + tid < HIDDEN_SIZE) {
          r_buf[bidy * TILE_HEIGHT * NUM_PARTIALS * HIDDEN_SIZE +
                k * NUM_PARTIALS * HIDDEN_SIZE + 
                bidx * HIDDEN_SIZE + 
                i * NUM_THREADS + tid] = r_dot_products[i];
        }
      }
    }
    
    // Synchronize between recurrent iterations - signal stage
    if (tid == 0) {
      syncIn[bidy * gridDim.x + bidx] = sequence_iteration + 1;
    }
    
    // Clear the output buffer
    for (int j = 0; j < TILE_HEIGHT; j++) {
      for (int i = 0; i < TILE_WIDTH; i++) {
        outputs_reg[j][i] = 0.f;
      }
    }
    
    // Populate time independent r value
    float precompute_r[TILE_HEIGHT][ELEMS_PER_THREAD];
    for (int j = 0; j < TILE_HEIGHT; j++) {
      for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        if ( i * NUM_THREADS + tid < HIDDEN_SIZE) {
          precompute_r[j][i] = precomputed_offset_r[j * HIDDEN_SIZE * GRU_GATES + i * NUM_THREADS + tid];
        }
      }
    }
    precomputed_offset_r += HIDDEN_SIZE * BATCH_SIZE * GRU_GATES;
    
    // Populate the other time indepedent gate inputs
    if (work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
      precompute = *precomputed_offset;
      precomputed_offset += BATCH_SIZE * HIDDEN_SIZE * GRU_GATES;
    }
    
    // Synchronize between recurrent iterations - spin stage
    __threadfence();

    if (bidx == 0) {
      if (tid < gridDim.x) {
        while ( syncIn[bidy * gridDim.x + tid] != sequence_iteration + 1) {
        }
      }

      __syncthreads();

      if (tid == 0) {
        syncOut[bidy] = sequence_iteration + 1;
      }
    } else {
      if (tid == 0) {
        while (syncOut[bidy] != sequence_iteration + 1) {
        }
      }
      __syncthreads();
    }
    
    // Load r gate partial dot products
    float r[TILE_HEIGHT][ELEMS_PER_THREAD][NUM_PARTIALS];
    for (int k = 0; k < TILE_HEIGHT; k++) {
      for (int i = 0; i < NUM_PARTIALS; i++) {
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
          if (j * NUM_THREADS + tid < HIDDEN_SIZE) {
              r[k][j][i] = r_buf[bidy * TILE_HEIGHT * NUM_PARTIALS * HIDDEN_SIZE +
                                 k * NUM_PARTIALS * HIDDEN_SIZE +
                                 j * NUM_THREADS + tid +
                                 i * HIDDEN_SIZE];
          }
        }
      }
    }
    
    // Load h_t-1 into shared memory
    if (sequence_iteration != 0) {
      for (int j = 0; j < TILE_HEIGHT; j++) {
        for (int i = 0; i < HIDDEN_SIZE; i+= NUM_THREADS) {
          if (i + tid < HIDDEN_SIZE) {
            h_tile[j][i + tid] = output[(sequence_iteration - 1) * HIDDEN_SIZE * BATCH_SIZE + (bidy * TILE_HEIGHT + j) * HIDDEN_SIZE + i + tid];
          }
        }
      }
    }
    
    __syncthreads(); 
    
    // Redundant calculate of r gate calculations (dot product, time independent, activation) and broadcast to shared memory 
    for (int k = 0; k < TILE_HEIGHT; k++) {
      for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        if (j * NUM_THREADS + tid < HIDDEN_SIZE) {
          float r_val = 0.f;
          for (int i = 0; i < NUM_PARTIALS; i++) {
            r_val += r[k][j][i];
          }
          r_val += bias_r[j];
          r_val += precompute_r[k][j];
          r_val = sigmoidf(r_val);
          r_val = r_val * h_tile[k][j * NUM_THREADS + tid];
          r_tile[k][j * NUM_THREADS + tid] = r_val;
        }
      }
    }
    
    __syncthreads();
     
    
    // R gate computation finished, so gates z and h_cand now perform tiled matrix multiplication
    // Note separate codepaths because compiler would otherwise introduce divergence
    if (g_type == 0) {
      for (int k = 0; k < LENGTH; k++) {
        for (int j = 0; j < TILE_HEIGHT; j++) {
          float val = r_tile[j][k * GROUP_THREADS + work_group.thread_rank()];
          for (int i = 0; i < TILE_WIDTH; i++) {
            outputs_reg[j][i] += weights_reg[i][k] * val;
          }
        }
      }
    } else {
      for (int k = 0; k < LENGTH; k++) {
        for (int j = 0; j < TILE_HEIGHT; j++) {
          float val = h_tile[j][k * GROUP_THREADS + work_group.thread_rank()];
          for (int i = 0; i < TILE_WIDTH; i++) {
            outputs_reg[j][i] += weights_reg[i][k] * val;
          }
        }
      }
    }

    // Reduction
    for (int j = 0; j < TILE_HEIGHT; j++) {
      for (int i = 0; i < TILE_WIDTH; i++) {
        for (int k = 1; k < GROUP_THREADS; k *= 2) {
          outputs_reg[j][i] += work_group.shfl_xor(outputs_reg[j][i], k);
        }
      }
    }
    
    // Gate activations
    if (work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
      int x = work_group.thread_rank() % TILE_WIDTH;
      int y = work_group.thread_rank() / TILE_WIDTH;

      float val = outputs_reg[y][x] + precompute + bias;
      
      if (g_type == 0) {
        val = sigmoidf(val);
        z_gate[y][wg_id * TILE_WIDTH + x] = val;
      } else {
        val = tanh(val);
        h_gate[y][wg_id * TILE_WIDTH + x] = val;
      }
    }

    __syncthreads();
    
    // Broadcast outputs
    if (tid < OUTPUT_TILE_WIDTH * TILE_HEIGHT) {
      int x = tid % OUTPUT_TILE_WIDTH;
      int y = tid / OUTPUT_TILE_WIDTH;
      if (bidx * OUTPUT_TILE_WIDTH + x < HIDDEN_SIZE) {
        float z_val = z_gate[y][x];
        float h_val = h_gate[y][x];
        float h_old_val = h_tile[y][bidx * OUTPUT_TILE_WIDTH + x];

        float out_val = (1 - z_val) * h_val + z_val * h_old_val;
        h_tile[y][bidx * OUTPUT_TILE_WIDTH + x] = out_val;
        output[sequence_iteration * HIDDEN_SIZE * BATCH_SIZE + (bidy * TILE_HEIGHT + y) * HIDDEN_SIZE + bidx * OUTPUT_TILE_WIDTH + x] = out_val;
      }
    }
    
    __syncthreads();

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
        output[j * hidden_size * GRU_GATES + k * GRU_GATES + i] = weights.at(i)[j * hidden_size + k];
      }
    }
  }

}

template<typename T>
void process_hidden_weights(T * output, std::vector<T*> weights, uint32_t hidden_size) {
  
  // For each output element
  for (uint32_t j = 0; j < hidden_size; j++) {
    // For gates z and h
    for (uint32_t k = 0; k < GRU_GATES - 1; k++) {
      // For each element for that gate
      for (uint32_t i = 0; i < hidden_size; i++) {
        // Indices 4 and 5 correspond to the z and h weights
        output[j * (GRU_GATES - 1) * hidden_size + k * hidden_size + i] = weights.at(4 + k)[i * hidden_size + j];
      }
    }
  }

}

template<typename T>
void process_biases(T * output, std::vector<T*> weights, uint32_t hidden_size) {
  int err = 0;
  // For each output element
  for (uint32_t k = 0; k < hidden_size; k++) {
    // Colocate the biases for each element
    for (uint32_t i = 0; i < GRU_GATES - 1; i++) {
      output[k * (GRU_GATES - 1) + i] = weights.at(i + 7)[k];
      if (weights.at(i + 7)[k] != 0.5) err++;
    }
  }

}

// Free buffers (all tiling dimension dependent)
template <typename T>
void GRULayerSingle<T>::reset() {
  cudaFreeHost((void *) this->packed_input_weights);
  cudaFreeHost((void *) this->packed_hidden_weights);
  cudaFreeHost((void *) this->packed_biases);
  cudaFree((void *) this->packed_hidden_weights_r_gpu);
  cudaFree((void *) this->packed_biases_r_gpu);
  cudaFree((void *) this->packed_input_weights_gpu);
  cudaFree((void *) this->packed_hidden_weights_gpu);
  cudaFree((void *) this->packed_biases_gpu);
}

// Initialize and fill trained parameter buffers
template<typename T>
uint32_t GRULayerSingle<T>::initialize() {
  
  uint32_t input_footprint = input_weight_footprint();
  uint32_t hidden_footprint = hidden_weight_footprint();
  uint32_t hidden_r_footprint = hidden_weight_r_footprint();
  uint32_t bias_footprint = bias_weight_footprint();
  uint32_t bias_r_footprint = bias_weight_r_footprint();
  
  // Allocate buffers
  cudaHostAlloc((void **) &(this->packed_input_weights), input_footprint, cudaHostAllocDefault); CUDA_ERR;
  cudaHostAlloc((void **) &(this->packed_hidden_weights), hidden_footprint, cudaHostAllocDefault); CUDA_ERR;
  cudaHostAlloc((void **) &(this->packed_biases), bias_footprint, cudaHostAllocDefault); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_input_weights_gpu), input_footprint); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_hidden_weights_gpu), hidden_footprint); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_biases_gpu), bias_footprint); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_hidden_weights_r_gpu), hidden_r_footprint); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_biases_r_gpu), bias_r_footprint); CUDA_ERR;
  
  // Reorganize weights
  process_input_weights(this->packed_input_weights, this->host_weights, this->input_size, this->hidden_size);
  process_hidden_weights(this->packed_hidden_weights, this->host_weights, this->hidden_size);
  process_biases(this->packed_biases, this->host_weights, this->hidden_size);

  // Send to GPU
  cudaMemcpy(this->packed_input_weights_gpu, this->packed_input_weights, input_footprint, cudaMemcpyHostToDevice); CUDA_ERR;
  cudaMemcpy(this->packed_hidden_weights_gpu, this->packed_hidden_weights, hidden_footprint, cudaMemcpyHostToDevice); CUDA_ERR;
  cudaMemcpy(this->packed_biases_gpu, this->packed_biases, bias_footprint, cudaMemcpyHostToDevice); CUDA_ERR;
  cudaMemcpy(this->packed_hidden_weights_r_gpu, this->host_weights.at(WEIGHTS_HIDDEN_R), hidden_r_footprint, cudaMemcpyHostToDevice); CUDA_ERR;
  cudaMemcpy(this->packed_biases_r_gpu, this->host_weights.at(BIAS_R), bias_r_footprint, cudaMemcpyHostToDevice); CUDA_ERR;

  return 0;

}

// Reset model parameters
template <typename T>
void GRUModelSingle<T>::reset() {

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

// Initialize model buffers
template <typename T>
uint32_t GRUModelSingle<T>::initialize() {
  
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
  this->gpu_weights_hidden_r = this->layers[0].get_packed_hidden_weights_r_gpu();
  this->gpu_biases_r = this->layers[0].get_packed_biases_r_gpu();
  
  this->mm_k = this->initial_input_size;
  this->mm_n = this->output_size * GRU_GATES;
  this->num_partials = (this->output_size + this->tile_width - 1) / this->tile_width;

  // Single sized output buffer (Will change for multi-layer networks, one output per iteration networks)
  cudaHostAlloc((void **) &(this->host_output), this->output_size * this->batch_size * sizeof(T), cudaHostAllocDefault);
  cudaMalloc((void **) &(this->gpu_output), this->output_size * this->batch_size * sizeof(T));
  
  // Assume batch size less than 200
  cudaMalloc((void **) &(this->gpu_inputs), this->initial_input_size * this->batch_size * 200 * sizeof(T));
  cudaMalloc((void **) &(this->gpu_r), this->output_size * this->batch_size * this->num_partials * sizeof(T));
  cudaMalloc((void **) &(this->gpu_precompute), this->output_size * this->batch_size * GRU_GATES * 200 * sizeof(T));

  // Hidden state initializer allocation
  cudaMalloc((void **) &(this->gpu_hidden_initializer), this->output_size * this->batch_size * sizeof(T));
  cudaMemset((void *)this->gpu_hidden_initializer, 0, this->output_size * this->batch_size * sizeof(T));

  cudaMalloc((void **) &(this->gpu_syncIn), 80 * sizeof(int));
  cudaMalloc((void **) &(this->gpu_syncOut), 80 * sizeof(int));
 
  this->paramsMM[0] = (void*) &(this->gpu_inputs);
  this->paramsMM[1] = (void*) &(this->gpu_weights_input);
  this->paramsMM[2] = (void*) &(this->gpu_precompute);
  this->paramsMM[4] = (void*) &(this->mm_k);
  this->paramsMM[5] = (void*) &(this->mm_n);

  this->paramsGRU[0] = (void*) &(this->gpu_precompute);
  this->paramsGRU[1] = (void*) &(this->gpu_hidden_initializer);
  this->paramsGRU[2] = (void*) &(this->gpu_weights_hidden_r);
  this->paramsGRU[3] = (void*) &(this->gpu_weights_hidden);
  this->paramsGRU[4] = (void*) &(this->gpu_biases_r);
  this->paramsGRU[5] = (void*) &(this->gpu_biases);
  this->paramsGRU[6] = (void*) &(this->gpu_r);
  this->paramsGRU[7] = (void*) &(this->gpu_output);
  this->paramsGRU[8] = (void*) &(this->gpu_syncIn);
  this->paramsGRU[9] = (void*) &(this->gpu_syncOut);

  return 0;

}

// Define tiling configuration (should be encapsulated elsewhere)
template <typename T>
void GRUModelSingle<T>::set_configuration(int x, int y, int g, int t) {
  this->tile_width = x;
  this->tile_height = y;
  this->num_groups = g;
  this->group_threads = t;
}

// Process input sequence batch
template <typename T>
float GRUModelSingle<T>::run_input(T* input, uint32_t * length) {
  
  // Define remaining kernel parameters
  this->mm_m = this->batch_size * *length;
  this->paramsMM[3] = (void *) &(this->mm_m);
  this->paramsGRU[10] = (void *) length;
  
  // GEMM Kernel dimensioning
  dim3 mm_grid = dim3((this->mm_n + MM_TILE_SIZE - 1) / MM_TILE_SIZE, (this->mm_m + MM_TILE_SIZE - 1) / MM_TILE_SIZE);
  dim3 mm_block = dim3(MM_BLOCK_SIZE, MM_BLOCK_SIZE);
  size_t mm_sm_requirement = MM_TILE_SIZE * MM_TILE_SIZE * 2 * sizeof(float);
  
  // GRU Kernel dimensioning
  int effective_w = (this->tile_width * this->num_groups) / 2;
  dim3 gru_rnn_grid = dim3((this->output_size + effective_w - 1) / effective_w, (this->batch_size + this->tile_height - 1) / this->tile_height);
  // While there are three gates, we use just two work groups per output to satisfy the dependency
  dim3 gru_rnn_block = dim3(this->num_groups * this->group_threads);
  unsigned block_size = gru_rnn_block.x;
  unsigned grid_size = gru_rnn_grid.x * gru_rnn_grid.y;

  // GRU Kernel instantiation (currently only configured for manual tuning)
  void * kernel = (void *)gru_rnn<256, 3, 1, 32, 8, 10>;
  
  // Check occupancy to prevent hangs
  int numBlocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, block_size, 0);
  if (grid_size > 80 * numBlocks) {
    printf("grid_size: %3d numBlocks: %3d\n", grid_size, numBlocks);
    return -std::numeric_limits<float>::infinity();
  }

  cudaEvent_t start, end;
  float elapsed;
  
  // Send sequence
  cudaMemcpy(this->gpu_inputs, input, this->initial_input_size * this->batch_size * *length * sizeof(T), cudaMemcpyHostToDevice);
  
  // Timing
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  
  // Kernel launches
  cudaLaunchKernel((void *)matmul, mm_grid, mm_block, this->paramsMM, mm_sm_requirement);
  cudaLaunchKernel(kernel, gru_rnn_grid, gru_rnn_block, this->paramsGRU);
  
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, start, end);
  
  cudaMemcpy(this->host_output, this->gpu_output, this->output_size * this->batch_size * sizeof(T), cudaMemcpyDeviceToHost);

#ifdef DEBUG
  // Value checking
  for (int i = 0; i < this->batch_size; i++) {
    printf("Sequence %2d\n", i);
    for (int j = 0; j < this->output_size; j++) {
      printf("%f ", this->host_output[i * this->output_size + j]);
    }
    printf("\n");
  }
  printf("\n");
#endif
  
  // Runtime error checking
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
template uint32_t GRULayerSingle<float>::initialize();
template uint32_t GRUModelSingle<float>::initialize();
template void GRULayerSingle<float>::reset();
template void GRUModelSingle<float>::reset();
template void GRUModelSingle<float>::set_configuration(int, int, int, int);
template float GRUModelSingle<float>::run_input(float *, uint32_t *);
