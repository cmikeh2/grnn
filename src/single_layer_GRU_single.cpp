// Runtime files
#include "GRU_single.h"
#include "misc.h"

// Other includes
#include <iostream>
#include <getopt.h>
#include <stdlib.h>
#include <vector>
#include <cassert>
#include <cstdio>
#include <cuda_profiler_api.h>


int main(int argc, char** argv) {

  uint32_t input = 256;
  uint32_t hidden = 256;
  uint32_t x_tile_size = 3;
  uint32_t y_tile_size = 1;
  uint32_t num_groups = 32;
  uint32_t group_threads = 8;
  uint32_t batch = 10;
  uint32_t input_length = 100;

  std::vector<float *> weights;
  create_dummy_weights_gru(weights, input, hidden);
 
  // Create layer
  GRULayerSingle<float> layer = GRULayerSingle<float>(input, hidden, batch, weights);
  
  // Declare model based on layer
  GRUModelSingle<float> model = GRUModelSingle<float>( {layer} );
  
  // Simple checks
  assert(input == model.get_initial_input_size());
  assert(batch == model.get_batch_size());
  assert(hidden == model.get_output_size());
  
  model.set_configuration(x_tile_size, y_tile_size, num_groups, group_threads);
  model.initialize();
  
  float * testInput;
  cudaHostAlloc((void **) &testInput, sizeof(float) * batch * input * input_length, cudaHostAllocDefault); CUDA_ERR;

  for (uint32_t i = 0; i < batch * input * input_length; i++) {
    testInput[i] = (float)(i % input) / (float)input;
  }

#ifdef DEBUG
  float temp = model.run_input(testInput, &input_length);
#else
  float time = 0.0f;
  for (int i = 0; i < 1000; i++) {
    float temp = model.run_input(testInput, &input_length);
  }
  cudaProfilerStart();
  for (int i = 0; i < 1000; i++) {
    float run_time = model.run_input(testInput, &input_length);
    time += run_time;
  }
  cudaProfilerStop();
  std::cout << time / 1000 << " ms\n";
#endif

  return 0;
}

