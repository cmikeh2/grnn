#ifndef GRUBASE_H
#define GRUBASE_H

#include "RNNBase.h"

#include <iostream>

#define WEIGHTS_INPUT_R 0
#define WEIGHTS_INPUT_Z 1
#define WEIGHTS_INPUT_H 2
#define WEIGHTS_HIDDEN_R 3
#define WEIGHTS_HIDDEN_Z 4
#define WEIGHTS_HIDDEN_H 5
#define BIAS_R 6
#define BIAS_Z 7
#define BIAS_H 8

#define GRU_GATES 3


template<typename T>
class GRULayerSingle : public RNNLayerBase<T> {
  
  private:
    T * packed_hidden_weights_r_gpu;
    T * packed_biases_r_gpu;

  public:
    GRULayerSingle(uint32_t i_s, uint32_t h_s, uint32_t b_s, std::vector<T*> l) :
      RNNLayerBase<T>(i_s, h_s, b_s, l) {}

    uint32_t initialize();
    void reset();

    // Total footprint of the input weights (makes initialize code cleaner)
    uint32_t input_weight_footprint() {
      return this->input_size * GRU_GATES * this->hidden_size * sizeof(T);
    }
    
    // Excludes intermediaries, used for data copying
    uint32_t hidden_weight_footprint() {
      return this->hidden_size * (GRU_GATES - 1) * this->hidden_size * sizeof(T);
    }

    // This function may need to be modified in order to avoid bank conflicts
    uint32_t bias_weight_footprint() {
      return this->hidden_size * (GRU_GATES - 1) * sizeof(T);
    }

    uint32_t hidden_weight_r_footprint() {
      return this->hidden_size * this->hidden_size * sizeof(T);
    }

    uint32_t bias_weight_r_footprint() {
      return this->hidden_size * sizeof(T);
    }

    T * get_packed_hidden_weights_r_gpu() {
      return this->packed_hidden_weights_r_gpu;
    }

    T * get_packed_biases_r_gpu() {
      return this->packed_biases_r_gpu;
    }
};

template<typename T>
class GRUModelSingle : public RNNBase<GRULayerSingle, T> {

  private:
    T * gpu_r;
    T * gpu_weights_hidden_r;
    T * gpu_biases_r;

    void * paramsGRU[11];
    int num_partials;
    
  public:
    GRUModelSingle(std::initializer_list< GRULayerSingle<T> > l) :
      RNNBase<GRULayerSingle, T>(l) {}
    
    void set_configuration(int x, int y, int g, int t);

    uint32_t initialize();
    void reset();

    float run_input(T* input, uint32_t * length);
};

#endif
