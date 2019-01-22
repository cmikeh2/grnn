#ifndef LSTMBASE_H
#define LSTMBASE_H

#include "RNNBase.h"

#include <iostream>

#define WEIGHTS_INPUT_F 0
#define WEIGHTS_INPUT_I 1
#define WEIGHTS_INPUT_C 2
#define WEIGHTS_INPUT_O 3
#define WEIGHTS_HIDDEN_F 4
#define WEIGHTS_HIDDEN_I 5
#define WEIGHTS_HIDDEN_C 6
#define WEIGHTS_HIDDEN_O 7
#define BIAS_F 8
#define BIAS_I 9
#define BIAS_C 10
#define BIAS_O 11

#define LSTM_GATES 4


template<typename T>
class LSTMLayer : public RNNLayerBase<T> {
  
  private:

  public:
    LSTMLayer(uint32_t i_s, uint32_t h_s, uint32_t b_s, std::vector<T*> l) :
      RNNLayerBase<T>(i_s, h_s, b_s, l) {}

    uint32_t initialize();
    void reset();

    // Total footprint of the input weights (makes initialize code cleaner)
    uint32_t input_weight_footprint() {
      return this->input_size * LSTM_GATES * this->hidden_size * sizeof(T);
    }
    
    // Excludes intermediaries, used for data copying
    uint32_t hidden_weight_footprint() {
      return this->hidden_size * LSTM_GATES * this->hidden_size * sizeof(T);
    }

    // This function may need to be modified in order to avoid bank conflicts
    uint32_t bias_weight_footprint() {
      return this->hidden_size * LSTM_GATES * sizeof(T);
    }

};

template<typename T>
class LSTMModel : public RNNBase<LSTMLayer, T> {

  private:
    void* paramsLSTM[8];
    
  public:
    LSTMModel(std::initializer_list< LSTMLayer<T> > l) :
      RNNBase<LSTMLayer, T>(l) {}

    void set_configuration(int x, int y, int g, int t);

    uint32_t initialize();
    void reset();

    float run_input(T* input, uint32_t * length);
};

#endif
