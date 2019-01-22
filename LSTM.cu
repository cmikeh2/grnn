#include "LSTM.h"
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
// a calculation on a 4x4 tile.

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
__global__ void lstm_rnn( const float* precomputed_inputs,
                          const float* hidden_initializer,
                          const float* weights, 
                          const float* biases, 
                          float* output,
                          volatile int* syncIn,
                          volatile int* syncOut,
                          uint32_t length) {
  
  // Indexing helpers
  int tid = threadIdx.x;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;
  int wg_id = tid / GROUP_THREADS;
  // LENGTH - How many weights for each gate output does a single thread need to store
  constexpr int LENGTH = (HIDDEN_SIZE + GROUP_THREADS - 1) / GROUP_THREADS;
  // BUFFER_SIZE - Number of elements to reserve in shared memory for each output. Effectively 
  // rounds up HIDDEN_SIZE to multiple of GROUP_THREADS
  constexpr int BUFFER_SIZE = LENGTH * GROUP_THREADS;
  // OUTPUT_TILE_WIDTH - How many full elements are produced by the threadblock. At scheduling time,
  // must ensure that launched configuration produces full elements within a single threadblock
  constexpr int OUTPUT_TILE_WIDTH = NUM_GROUPS * TILE_WIDTH / LSTM_GATES;
  
  // Static shared memory allocation
  __shared__ float hidden_tile[TILE_HEIGHT][BUFFER_SIZE];
  __shared__ float cell_state[TILE_HEIGHT][OUTPUT_TILE_WIDTH];
  __shared__ float forget_gate[TILE_HEIGHT][OUTPUT_TILE_WIDTH];
  __shared__ float input_gate[TILE_HEIGHT][OUTPUT_TILE_WIDTH];
  __shared__ float cand_gate[TILE_HEIGHT][OUTPUT_TILE_WIDTH];
  __shared__ float out_gate[TILE_HEIGHT][OUTPUT_TILE_WIDTH];
  
  // Weights in the register file
  float weights_reg[TILE_WIDTH][LENGTH];
  float outputs_reg[TILE_HEIGHT][TILE_WIDTH];
  float bias = 0.0f;
  float precompute = 0.0f;

  // Cooperative group helpers
  thread_block bl = this_thread_block();
  thread_block_tile<GROUP_THREADS> work_group = tiled_partition<GROUP_THREADS>(bl);

  // Tile width is the number of gate outputs produce by a single warp
  for (int i = 0; i < TILE_WIDTH; i++) {
    // Global gate id for fetching weights. 
    // bidx * TILE_WIDTH * NUM_GROUPS -> first gate index processed by the threadblock
    // (tid / GROUP_THREADS) * TILE_WIDTH -> first gate index within processed by a warp within the threadblock
    // i -> current gate index within the warp's assigned gates
    int gate_id = bidx * TILE_WIDTH * NUM_GROUPS + wg_id * TILE_WIDTH + i;
    // HIDDEN_SIZE * LSTM_GATES -> number of total gates that need to be computed
    if (gate_id < HIDDEN_SIZE * LSTM_GATES) {
      for (int j = 0; j < LENGTH; j++) {
        // Better to fully populate and check weight bounds once at loading than during each computation.
        if (j * GROUP_THREADS + work_group.thread_rank() < HIDDEN_SIZE) {
          weights_reg[i][j] = weights[gate_id * HIDDEN_SIZE + j * GROUP_THREADS + work_group.thread_rank()];
        } else {
          weights_reg[i][j] = 0.0f;
        }
      }
    }
  }

  // Assigns correct bias value to specific output. Prunes to only ensure that values are fetched that are necessary for later
  // for later computation
  if (work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
    if ((bidx * TILE_WIDTH * NUM_GROUPS + wg_id * TILE_WIDTH + work_group.thread_rank() % TILE_WIDTH) < HIDDEN_SIZE * LSTM_GATES) {
      bias = biases[bidx * TILE_WIDTH * NUM_GROUPS + wg_id * TILE_WIDTH + work_group.thread_rank() % TILE_WIDTH];
    } else {
      bias = 0;
    }
  }

  // Zero initialize the cell state
  if (tid < TILE_HEIGHT * OUTPUT_TILE_WIDTH) {
    cell_state[tid / OUTPUT_TILE_WIDTH][tid % OUTPUT_TILE_WIDTH] = 0.0f;
  }
  
  // Initialize hidden state buffer according to input / zero out rest of buffer
  for (int j = 0; j < TILE_HEIGHT; j++) {
    for (int i = 0; i < BUFFER_SIZE; i += NUM_GROUPS * GROUP_THREADS) {
      if (i + tid < HIDDEN_SIZE) {
        hidden_tile[j][i + tid] = hidden_initializer[(bidy * TILE_HEIGHT + j) * HIDDEN_SIZE + i + tid];
      } else if (i + tid < BUFFER_SIZE) {
        hidden_tile[j][i + tid] = 0.0f;
      }
    }
  }
  __syncthreads();
  
  // Zero dot product accumulators
  #pragma unroll
  for (int j = 0; j < TILE_HEIGHT; j++) {
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH; i++) {
      outputs_reg[j][i] = 0.0f;
    }
  }
  
  // Load first time independent values
  if ((bidx * TILE_WIDTH * NUM_GROUPS + wg_id * TILE_WIDTH + work_group.thread_rank() % TILE_WIDTH < HIDDEN_SIZE *  LSTM_GATES)
       && work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
    precompute = precomputed_inputs[bidy * TILE_HEIGHT * HIDDEN_SIZE * LSTM_GATES +
                                    bidx * TILE_WIDTH * NUM_GROUPS +
                                    wg_id * TILE_WIDTH +
                                    work_group.thread_rank() % TILE_WIDTH + 
                                    (work_group.thread_rank() / TILE_WIDTH) * HIDDEN_SIZE * LSTM_GATES];

  }
  
  // Loop for each iteration of the sequence length
  for (int sequence_iteration = 0; sequence_iteration < length; sequence_iteration++) {
    
    // Dot products
    #pragma unroll
    for (int k = 0; k < LENGTH; k++) {
      #pragma unroll
      for (int j = 0; j < TILE_HEIGHT; j++) {
        float val = hidden_tile[j][k * GROUP_THREADS + work_group.thread_rank()];
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
          outputs_reg[j][i] += weights_reg[i][k] * val;
        }
      }
    }
    
    // Reductions
    #pragma unroll
    for (int k = 1; k < GROUP_THREADS; k *= 2) {
      #pragma unroll
      for (int j = 0; j < TILE_HEIGHT; j++) {
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
          outputs_reg[j][i] += work_group.shfl_xor(outputs_reg[j][i], k);
        }
      }
    }
    
    // Remap work and compute activations
    if (work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
      int reg_y = work_group.thread_rank() / TILE_WIDTH;
      int reg_x = work_group.thread_rank() % TILE_WIDTH;
      float val = outputs_reg[reg_y][reg_x] + bias + precompute;

      int gate_id = (wg_id * TILE_WIDTH + work_group.thread_rank() % TILE_WIDTH) % LSTM_GATES;

      if (gate_id != 2) {
        val = sigmoidf(val);
      } else {
        val = tanhf(val);
      }
      int out_id = (wg_id * TILE_WIDTH + work_group.thread_rank() % TILE_WIDTH) / LSTM_GATES;
      if (gate_id == 0) {
        forget_gate[reg_y][out_id] = val;
      } else if (gate_id == 1) {
        input_gate[reg_y][out_id] = val;
      } else if (gate_id == 2) {
        cand_gate[reg_y][out_id] = val;
      } else {
        out_gate[reg_y][out_id] = val;
      }
    }

    // Synchronization enforces all intermediates are calculated before the data is shared across threads
    // for the elementwise operations.
    __syncthreads();
    
    int x = tid  % OUTPUT_TILE_WIDTH; 
    int y = tid  / OUTPUT_TILE_WIDTH;
    
    // Elementwise operations
    if (tid < OUTPUT_TILE_WIDTH * TILE_HEIGHT  &&
        (bidx * OUTPUT_TILE_WIDTH + x) < HIDDEN_SIZE &&
        (bidy * TILE_HEIGHT + y) < BATCH_SIZE) { 
      // Calculates the new cell state
      float cell_reg = cell_state[y][x] * forget_gate[y][x] + input_gate[y][x] * cand_gate[y][x];
      // Calculates the new output
      float out_reg = tanhf(cell_reg) * out_gate[y][x];

      // No synchronization necessary between the read and writes of cell state because it is guaranteed that only the
      // same thread will read/write to the element.
      cell_state[y][x] = cell_reg;

      // Broadcast output to global memory
      output[sequence_iteration * HIDDEN_SIZE * BATCH_SIZE + (bidy * TILE_HEIGHT + y) * HIDDEN_SIZE + bidx * OUTPUT_TILE_WIDTH + x] = out_reg;
    }
    
    // Escape recurrent loop when full sequence has been processed
    if (sequence_iteration + 1 == length) break;
    

    
    // Synchronize between recurrent iterations - signal stage
    if (tid == 0 ) {
      syncIn[(bidy * gridDim.x + bidx)] = sequence_iteration + 1;
    }
    __threadfence();
    
    // Zero the dot product accumulators
    #pragma unroll
    for (int j = 0; j < TILE_HEIGHT; j++) {
      #pragma unroll
      for (int i = 0; i < TILE_WIDTH; i++) {
        outputs_reg[j][i] = 0.0f;
      }
    }
    
    // Read precomputed value from memory (Since this is a read-only operation that does ot
    // use a shared intermediate, this can go before the memory barrier without correctness issues
    // Ideally, this will hide some latency, but needs profiling
    if ((bidx * TILE_WIDTH * NUM_GROUPS + wg_id * TILE_WIDTH + work_group.thread_rank() % TILE_WIDTH < HIDDEN_SIZE *  LSTM_GATES)
         && work_group.thread_rank() < TILE_WIDTH * TILE_HEIGHT) {
      precompute = precomputed_inputs[bidy * TILE_HEIGHT * HIDDEN_SIZE * LSTM_GATES +
                                      bidx * TILE_WIDTH * NUM_GROUPS +
                                      wg_id * TILE_WIDTH +
                                      work_group.thread_rank() % TILE_WIDTH + 
                                      (work_group.thread_rank() / TILE_WIDTH) * HIDDEN_SIZE * LSTM_GATES +
                                      (sequence_iteration + 1) * BATCH_SIZE * HIDDEN_SIZE * LSTM_GATES];

    }
    
    // Synchronize between recurrent iterations - spin stage
    if (bidx == 0) {
      if (tid < gridDim.x) {
        while (syncIn[(bidy * gridDim.x + tid)]  != sequence_iteration + 1) {
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
    
    // Load the hidden state into the input buffer in shared memory (coalesced)
    // Tile height * REDUCTION_WIDTH * SEQUENTIAL_LENGTH is equivalent to the tile height by hidden size
    // Reduction_width * TILE_WIDTH * LSTM_GATES is the number of threads launched (allows for loop unrolling)
    #pragma unroll
    for (int i = 0; i < TILE_HEIGHT; i++) {
      if (i + bidy * TILE_HEIGHT < BATCH_SIZE) {
        #pragma unroll
        for (int j = 0; j < HIDDEN_SIZE; j += NUM_GROUPS * GROUP_THREADS) {
          if (j + tid < HIDDEN_SIZE) {
            hidden_tile[i][j+tid] = output[sequence_iteration * HIDDEN_SIZE * BATCH_SIZE + (bidy * TILE_HEIGHT + i) * HIDDEN_SIZE + j + tid];
          } else if (j + tid < BUFFER_SIZE) {
            hidden_tile[i][j+tid] = 0.0f;
          }
        }
      }
    }

    // Enforce loading of data to shared memory before computation
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
      for (uint32_t i = 0; i < LSTM_GATES; i++) {
        output[(j * hidden_size + k) * LSTM_GATES + i] = weights.at(i)[j * hidden_size + k];
      }
    }
  }
}

template<typename T>
void process_hidden_weights(T * output, std::vector<T*> weights, uint32_t hidden_size) {
  
  // For each output element
  for (uint32_t j = 0; j < hidden_size; j++) {
    // For each gate
    for (uint32_t k = 0; k < LSTM_GATES; k++) {
      // For each element for that gate
      for (uint32_t i = 0; i < hidden_size; i++) {
        output[j * LSTM_GATES * hidden_size + k * hidden_size + i] = weights.at(4 + k)[i * hidden_size + j];
      }
    }
  }
}

template<typename T>
void process_biases(T * output, std::vector<T*> weights, uint32_t hidden_size) {

  // For each output element
  for (uint32_t k = 0; k < hidden_size; k++) {
    // Colocate the biases for each element
    for (uint32_t i = 0; i < LSTM_GATES; i++) {
      output[k * LSTM_GATES + i] = weights.at(i + 8)[k];
    }
  }
}

// Initialize all layer weights and send to GPU
template<typename T>
uint32_t LSTMLayer<T>::initialize() {
  
  uint32_t input_footprint = input_weight_footprint();
  uint32_t hidden_footprint = hidden_weight_footprint();
  uint32_t bias_footprint = bias_weight_footprint();
  
  // Weight buffer allocations
  cudaHostAlloc((void **) &(this->packed_input_weights), input_footprint, cudaHostAllocDefault); CUDA_ERR;
  cudaHostAlloc((void **) &(this->packed_hidden_weights), hidden_footprint, cudaHostAllocDefault); CUDA_ERR;
  cudaHostAlloc((void **) &(this->packed_biases), bias_footprint, cudaHostAllocDefault); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_input_weights_gpu), input_footprint); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_hidden_weights_gpu), hidden_footprint); CUDA_ERR;
  cudaMalloc((void **) &(this->packed_biases_gpu), bias_footprint); CUDA_ERR;
  
  // Reorganize weights (typically a transpose)
  process_input_weights(this->packed_input_weights, this->host_weights, this->input_size, this->hidden_size);
  process_hidden_weights(this->packed_hidden_weights, this->host_weights, this->hidden_size);
  process_biases(this->packed_biases, this->host_weights, this->hidden_size);
  
  // Send weights to GPU
  cudaMemcpy(this->packed_input_weights_gpu, this->packed_input_weights, input_footprint, cudaMemcpyHostToDevice); CUDA_ERR;
  cudaMemcpy(this->packed_hidden_weights_gpu, this->packed_hidden_weights, hidden_footprint, cudaMemcpyHostToDevice); CUDA_ERR;
  cudaMemcpy(this->packed_biases_gpu, this->packed_biases, bias_footprint, cudaMemcpyHostToDevice); CUDA_ERR;

  return 0;

}

// Free all allocated buffers. Only needed for full sweep benchmarking.
template <typename T>
void LSTMLayer<T>::reset() {
  cudaFreeHost((void *) this->packed_input_weights);
  cudaFreeHost((void *) this->packed_hidden_weights);
  cudaFreeHost((void *) this->packed_biases);
  cudaFree((void *) this->packed_input_weights_gpu);
  cudaFree((void *) this->packed_hidden_weights_gpu);
  cudaFree((void *) this->packed_biases_gpu);
}

// Allocate input/output buffers for the layer. Currently set up for only single layer models, but can be extended to multi-layer 
// without dramatic refactoring
template <typename T>
uint32_t LSTMModel<T>::initialize() {
  
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
  this->mm_n = this->output_size * LSTM_GATES;

  // Output allocations, assumes sequence length less than 100
  cudaHostAlloc((void **) &(this->host_output), this->output_size * this->batch_size * sizeof(T) * 100, cudaHostAllocDefault);
  cudaMalloc((void **) &(this->gpu_output), this->output_size * this->batch_size * sizeof(T) * 100);
  
  // Input allocations, assumes sequence length less than 100
  cudaMalloc((void **) &(this->gpu_inputs), this->initial_input_size * this->batch_size * 100 * sizeof(T));
  cudaMalloc((void **) &(this->gpu_precompute), this->output_size * this->batch_size * LSTM_GATES * 100 * sizeof(T));

  // Initialize hidden state, for our purposes we use 0's
  cudaMalloc((void **) &(this->gpu_hidden_initializer), this->output_size * this->batch_size * sizeof(T));
  cudaMemset((void *)this->gpu_hidden_initializer, 0, this->output_size * this->batch_size * sizeof(T));
  
  // Synchronization buffers. Always allocated to full dimensionality so that they may be easily reused from run to run
  cudaMalloc((void **) &(this->gpu_syncIn), 80 * sizeof(int) * LINE_SIZE);
  cudaMalloc((void **) &(this->gpu_syncOut), 80 * sizeof(int) * LINE_SIZE);
 
  // GEMM Kernel parameters
  this->paramsMM[0] = (void*) &(this->gpu_inputs);
  this->paramsMM[1] = (void*) &(this->gpu_weights_input);
  this->paramsMM[2] = (void*) &(this->gpu_precompute);
  this->paramsMM[4] = (void*) &(this->mm_k);
  this->paramsMM[5] = (void*) &(this->mm_n);

  // LSTM Kernel parameters
  this->paramsLSTM[0] = (void*) &(this->gpu_precompute);
  this->paramsLSTM[1] = (void*) &(this->gpu_hidden_initializer);
  this->paramsLSTM[2] = (void*) &(this->gpu_weights_hidden);
  this->paramsLSTM[3] = (void*) &(this->gpu_biases);
  this->paramsLSTM[4] = (void*) &(this->gpu_output);
  this->paramsLSTM[5] = (void*) &(this->gpu_syncIn);
  this->paramsLSTM[6] = (void*) &(this->gpu_syncOut);

  return 0;
}

// Frees model buffers
template <typename T>
void LSTMModel<T>::reset() {

  for (auto& l: this->layers) {
    l.reset();
  }

  cudaFreeHost((void *) this->host_output);
  cudaFree((void *) this->gpu_output);

  cudaFree((void *) this->gpu_inputs);
  cudaFree((void *) this->gpu_precompute);
}

// Defines tiling configuration (should be encapsulated elsewhere in the future)
template <typename T>
void LSTMModel<T>::set_configuration(int x, int y, int g, int t) {
  this->tile_width = x;
  this->tile_height = y;
  this->num_groups = g;
  this->group_threads = t;
}

// Processes input sequence (both independent and dependent)
template <typename T>
float LSTMModel<T>::run_input(T* input, uint32_t * length) {
  
  // Define remaining kernel parameters (primarily dependent on sequence length)
  this->mm_m = this->batch_size * *length;
  this->paramsMM[3] = (void *) &(this->mm_m);
  this->paramsLSTM[7] = (void *) length;
  
  // GEMM Kernel dimensioning
  dim3 mm_grid = dim3((this->mm_n + MM_TILE_SIZE - 1) / MM_TILE_SIZE, (this->mm_m + MM_TILE_SIZE - 1) / MM_TILE_SIZE);
  dim3 mm_block = dim3(MM_BLOCK_SIZE, MM_BLOCK_SIZE);
  size_t mm_sm_requirement = MM_TILE_SIZE * MM_TILE_SIZE * 2 * sizeof(float);
  
  // LSTM Kernel dimensioning
  int effective_w = (this->num_groups * this->tile_width) / LSTM_GATES;
  dim3 lstm_rnn_grid = dim3((this->output_size + effective_w - 1) / effective_w, (this->batch_size + this->tile_height - 1) / this->tile_height);
  dim3 lstm_rnn_block = dim3(this->num_groups * this->group_threads);  
  unsigned block_size = lstm_rnn_block.x;
  unsigned grid_size = lstm_rnn_grid.x * lstm_rnn_grid.y;

  // Kernel instantiation (currently configured for manual application of parameters)
  void * kernel = (void*)lstm_rnn<256, 2, 4, 64, 8, 40>;
  int numBlocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, block_size, 0);
   
  // Check occupancy prior to launch to prevent program hangs
  if (numBlocks == 0 || grid_size > 80) {
    printf("numBlocks: %2d grid_size: %3d, block_size: %3d\n", numBlocks, grid_size, block_size);
    return -std::numeric_limits<float>::infinity();
  }

  cudaEvent_t start, end;
  float elapsed;
  
  cudaMemcpy(this->gpu_inputs, input, this->initial_input_size * this->batch_size * *length * sizeof(T), cudaMemcpyHostToDevice);
  
  // Timing info
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  
  // Kernel launches
  cudaLaunchKernel((void *)matmul, mm_grid, mm_block, this->paramsMM, mm_sm_requirement);
  cudaLaunchKernel(kernel, lstm_rnn_grid, lstm_rnn_block, this->paramsLSTM);
  
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
  
  // Check for runtime errors
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
template uint32_t LSTMLayer<float>::initialize();
template uint32_t LSTMModel<float>::initialize();
template void LSTMModel<float>::set_configuration(int, int, int, int);
template float LSTMModel<float>::run_input(float *, uint32_t *);
template void LSTMModel<float>::reset();
template void LSTMLayer<float>::reset();
