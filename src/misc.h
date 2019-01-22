#ifndef MISC_H
#define MISC_H

#include <vector>
#include <iostream>
#include <stdio.h>
#include <cassert>
#include <fstream>

#define LINE_SIZE 1

#define CUDA_ERR {                                                                          \
  cudaError_t err;                                                                          \
  if ((err = cudaGetLastError()) != cudaSuccess) {                                         \
    printf("CUDA error: %d : %s : %s, line %d\n", err, cudaGetErrorString(err), __FILE__, __LINE__);  \
    exit(1);                                                                                \
  }                                                                                         \
}

#define MAX_SMEM 98304

template<typename T>
void create_dummy_weights_lstm(std::vector<T *> &weights, uint32_t input, uint32_t hidden) {
  // DUMMY INPUT WEIGHTS
  weights.push_back((T *)malloc(sizeof(T) * input * hidden));
  weights.push_back((T *)malloc(sizeof(T) * input * hidden));
  weights.push_back((T *)malloc(sizeof(T) * input * hidden));
  weights.push_back((T *)malloc(sizeof(T) * input * hidden));
  
  // DUMMY HIDDEN WEIGHTS
  weights.push_back((T *)malloc(sizeof(T) * hidden * hidden));
  weights.push_back((T *)malloc(sizeof(T) * hidden * hidden));
  weights.push_back((T *)malloc(sizeof(T) * hidden * hidden));
  weights.push_back((T *)malloc(sizeof(T) * hidden * hidden));

  // DUMMY BIASES
  weights.push_back((T *)malloc(sizeof(T) * hidden));
  weights.push_back((T *)malloc(sizeof(T) * hidden));
  weights.push_back((T *)malloc(sizeof(T) * hidden));
  weights.push_back((T *)malloc(sizeof(T) * hidden));

  uint32_t i, j;
  for (i = 0; i < 4; i++) {
    for (j = 0; j < input * hidden; j++) {
    weights.at(i)[j] = 1 / 1024.;
    }
  }

  for (i = 4; i < 8; i++) {
    for (j = 0; j < hidden * hidden; j++) {
    weights.at(i)[j] = 1 / 1024.;
    }
  }
  
  for (i = 8; i < 12; i++) {
    for (j = 0; j < hidden; j++) {
    weights.at(i)[j] = 0.5;
    }
  }
}

template<typename T>
void create_dummy_weights_gru(std::vector<T *> &weights, uint32_t input, uint32_t hidden) {
  // DUMMY INPUT WEIGHTS
  weights.push_back((T *)malloc(sizeof(T) * input * hidden));
  weights.push_back((T *)malloc(sizeof(T) * input * hidden));
  weights.push_back((T *)malloc(sizeof(T) * input * hidden));
  
  // DUMMY HIDDEN WEIGHTS
  weights.push_back((T *)malloc(sizeof(T) * hidden * hidden));
  weights.push_back((T *)malloc(sizeof(T) * hidden * hidden));
  weights.push_back((T *)malloc(sizeof(T) * hidden * hidden));

  // DUMMY BIASES
  weights.push_back((T *)malloc(sizeof(T) * hidden));
  weights.push_back((T *)malloc(sizeof(T) * hidden));
  weights.push_back((T *)malloc(sizeof(T) * hidden));

  uint32_t i, j;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < input * hidden; j++) {
    weights.at(i)[j] = 1./256.;
    }
  }

  for (i = 3; i < 6; i++) {
    for (j = 0; j < hidden * hidden; j++) {
    weights.at(i)[j] = 1./256.;
    }
  }
  
  for (i = 7; i < 9; i++) {
    for (j = 0; j < hidden; j++) {
    weights.at(i)[j] = 0.5;
    }
  }
}

#endif
