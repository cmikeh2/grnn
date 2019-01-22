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
class GRULayerDouble : public RNNLayerBase<T> {
  
  private:

  public:
    GRULayerDouble(uint32_t i_s, uint32_t h_s, uint32_t b_s, std::vector<T*> l) :
      RNNLayerBase<T>(i_s, h_s, b_s, l) {}

    uint32_t initialize();
    void reset();

    // Total footprint of the input weights (makes initialize code cleaner)
    uint32_t input_weight_footprint() {
      return this->input_size * GRU_GATES * this->hidden_size * sizeof(T);
    }
    
    // Excludes intermediaries, used for data copying
    uint32_t hidden_weight_footprint() {
      return this->hidden_size * GRU_GATES * this->hidden_size * sizeof(T);
    }

    // This function may need to be modified in order to avoid bank conflicts
    uint32_t bias_weight_footprint() {
      return this->hidden_size * GRU_GATES * sizeof(T);
    }
};

template<typename T>
class GRUModelDouble : public RNNBase<GRULayerDouble, T> {

  private:
    T * gpu_r;

    void * paramsGRU[9];
    
  public:
    GRUModelDouble(std::initializer_list< GRULayerDouble<T> > l) :
      RNNBase<GRULayerDouble, T>(l) {}

    void set_configuration(int x, int y, int g, int t);

    uint32_t initialize();
    void reset();

    float run_input(T* input, uint32_t * length);
};

#endif
