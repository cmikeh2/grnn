# Compiler parameters
CXX := nvcc
CPPFLAGS := -O3 -std=c++11
CUDAFLAGS := -arch=compute_70 -code=sm_70 -D_FORCE_INLINES --ptxas-options='-v -warn-lmem-usage -warn-spills' --nvlink-options='-v'
DEBUGFLAGS := -D DEBUG

# Rules
all: run

### Regular compilation rules
bin/LSTM.o: LSTM.cu LSTM.h RNNBase.h misc.h
	$(CXX) -c $< -o $@ $(CPPFLAGS) $(CUDAFLAGS)

LSTM: single_layer_LSTM.cpp LSTM.h RNNBase.h misc.h bin/LSTM.o
	$(CXX) $< bin/LSTM.o -o bin/$@ $(CPPFLAGS) $(CUDAFLAGS)

bin/GRU_single.o: GRU_single.cu GRU_single.h RNNBase.h misc.h
	$(CXX) -c $< -o $@ $(CPPFLAGS) $(CUDAFLAGS)

GRU_single: single_layer_GRU_single.cpp GRU_single.h RNNBase.h misc.h bin/GRU_single.o
	$(CXX) $< bin/GRU_single.o -o bin/$@ $(CPPFLAGS) $(CUDAFLAGS)

bin/GRU_double.o: GRU_double.cu GRU_double.h RNNBase.h misc.h
	$(CXX) -c $< -o $@ $(CPPFLAGS) $(CUDAFLAGS)

GRU_double: single_layer_GRU_double.cpp GRU_double.h RNNBase.h misc.h bin/GRU_double.o
	$(CXX) $< bin/GRU_double.o -o bin/$@ $(CPPFLAGS) $(CUDAFLAGS)

clean:
	rm bin/*
