#ifndef RNNBASE_H
#define RNNBASE_H

#include "misc.h"

#include <vector>
#include <initializer_list>
#include <iostream>
#include <cuda_runtime_api.h>

template<typename T>
class RNNLayerBase {
  
  protected:
    //LAYER TOPOLOGY
    uint32_t hidden_size;
    uint32_t input_size;
    uint32_t batch_size;
    uint32_t block_width;

    // UNMODIFIED HOST WEIGHTS (SHOULD BE POINTERS ON HEAP)
    std::vector<T*> host_weights;

    // WEIGHTS PACKED INTO SUITABLE CONFIGURATION FOR SHARED MEM
    T * packed_input_weights;
    T * packed_input_weights_gpu;
    T * packed_hidden_weights;
    T * packed_hidden_weights_gpu;
    T * packed_biases;
    T * packed_biases_gpu;
    
  public:
    RNNLayerBase(uint32_t i_s, uint32_t h_s, uint32_t b_s, std::vector<T*> l) :
      input_size(i_s),
      hidden_size(h_s),
      batch_size(b_s),
      host_weights(l) {
    }

    // PACKS WEIGHTS FOR SHARED MEMORY TRANSFER
    virtual uint32_t initialize() =0;
    virtual void reset() =0;
    
    virtual uint32_t input_weight_footprint() =0;
    virtual uint32_t hidden_weight_footprint() =0;
    virtual uint32_t bias_weight_footprint() = 0;

    // SETTERS
    void set_block_width(uint32_t width) { block_width = width; }
    
    // GETTERS (CHANGING SIZE OF LAYER NOT SUPPORTED)
    uint32_t get_hidden_size() { return hidden_size; }
    uint32_t get_input_size() { return input_size; }
    uint32_t get_batch_size() { return batch_size; }
    T * get_packed_input_weights_gpu() { return packed_input_weights_gpu; }
    T * get_packed_hidden_weights_gpu() { return packed_hidden_weights_gpu; }
    T * get_packed_biases_gpu() { return packed_biases_gpu; }

};

template<template<typename> typename L, typename T>
class RNNBase {

  protected:
    // Vector of layers
    std::vector< L<T> > layers;

    // Topology
    uint32_t initial_input_size;
    uint32_t batch_size;
    uint32_t output_size;
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t num_groups;
    uint32_t group_threads;
    uint32_t mm_m;
    uint32_t mm_n;
    uint32_t mm_k;
    
    // Data
    T * gpu_inputs;
    T * gpu_hidden_initializer;
    T * gpu_weights_input;
    T * gpu_weights_hidden;
    T * gpu_biases;
    T * gpu_precompute;
    T * gpu_output;
    int * gpu_syncIn;
    int * gpu_syncOut;
    T * host_output;

    // Kernel Parameters
    void* paramsMM[6];

  public:
    RNNBase(std::initializer_list< L<T> > l) : layers(l) {
      this->initial_input_size = layers.front().get_input_size();
      this->batch_size = layers.front().get_batch_size();
      this->output_size = layers.back().get_hidden_size();
    }

    virtual uint32_t initialize() =0;
    virtual void reset() =0;
    
    // Transfers input to the GPU, runs kernel, fetches output
    virtual float run_input(T * input, uint32_t * length) =0;
  
    // Configure tiling parameters
    virtual void set_configuration(int x, int y, int g, int t) =0;
    
    // GETTERS
    uint32_t get_initial_input_size() { return initial_input_size; }
    uint32_t get_batch_size() { return batch_size; }
    uint32_t get_output_size() { return output_size; }
    
};

#endif
