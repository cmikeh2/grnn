#!/usr/bin/python3

from enum import Enum
from math import ceil
from math import floor
from math import sqrt
from math import log2
import argparse
import itertools
import numpy as np

sms = 80
max_regs_thread = 224
max_regs_sm = 65536
max_threads = 1024
fma_cost = 4
l2_bus = 64
l2_lat = 50
l1_lat = 5
warp_size = 32

class ModelType(Enum):
  LSTM = 0
  GRU = 1

model_gates = {ModelType.LSTM : 4, ModelType.GRU : 2}
model_scale = {ModelType.LSTM : 1, ModelType.GRU : 1.5} ## Weights stored per work group
string_model = {'LSTM' : ModelType.LSTM, 'GRU' : ModelType.GRU}
model_string = {ModelType.LSTM : 'LSTM', ModelType.GRU : 'GRU'}

class ModelConfig:

  def __init__(self, mt, hs, bs, tw, th, nwg, rw, sy=None):
    self.model_type = mt
    self.hidden_size = hs
    self.batch_size = bs
    self.tile_width = tw
    self.tile_height = th
    self.reduction_width = rw
    self.num_work_groups = nwg
    self.sub_tile_width = ceil(self.tile_width * model_gates[self.model_type] / self.num_work_groups)
    self.num_threads = self.reduction_width * self.num_work_groups
    self.num_SMs = ceil(self.hidden_size / self.tile_width) * ceil(self.batch_size / self.tile_height)
    if self.model_type is ModelType.GRU:
      self.sync = sy
      if self.sync is 1:
        self.weights_per_thread = self.sub_tile_width * self.hidden_size / self.reduction_width + self.tile_width * ceil(self.hidden_size / self.num_threads)
      else:
        self.weights_per_thread = self.sub_tile_width * self.hidden_size / self.reduction_width * 1.5
    else:
      self.weights_per_thread = self.sub_tile_width * self.hidden_size / self.reduction_width
    self.cost = self.fitness()

  def __str__(self):
    rep = "Model Config:\n"
    rep += "\tModel Info:\n"
    rep += "\t\tModel Type: " + str(self.model_type) +"\n"
    rep += "\t\tHidden Size: " + str(self.hidden_size) +"\n"
    rep += "\t\tBatch Size: " + str(self.batch_size) +"\n"
    rep += "\tConfiguration Parameters:\n"
    rep += "\t\tTile Width: " + str(self.tile_width) +"\n"
    rep += "\t\tTile Height: " + str(self.tile_height) +"\n"
    rep += "\t\tReduction Width: " + str(self.reduction_width) +"\n"
    rep += "\t\tNum Work Groups: " + str(self.num_work_groups) +"\n"
    rep += "\tOccupancy Metrics:\n"
    rep += "\t\tNumber of SMs: " + str(self.num_SMs) +"\n"
    rep += "\t\tWeights Per SM: " + str(self.tile_width * self.hidden_size * model_gates[self.model_type] * model_scale[self.model_type]) +"\n"
    rep += "\t\tSub Tile Width: " + str(self.sub_tile_width) +"\n"
    rep += "\t\tWeights Per Threads: " + str(self.weights_per_thread) +"\n"
    rep += "\tFitness: " + str(self.cost) + "\n"
    return rep

  def is_valid(self):
    if self.num_SMs > 80:
      return False
    elif self.weights_per_thread > max_regs_thread:
      return False
    elif (self.weights_per_thread + 32) * self.num_threads > max_regs_sm:
      return False
    elif self.sub_tile_width * self.tile_height > self.reduction_width:
      return False
    elif self.num_threads > max_threads:
      return False
    elif (model_gates[self.model_type] * self.tile_width % self.num_work_groups) is not 0:
      return False
#   elif self.num_threads % 128 is not 0:
#     return False
    else:
      return True
  
  def fma_heuristic(self):
    sequential_length = self.hidden_size / self.reduction_width
    self.partition_occupancy = ceil(self.num_threads / 32 / 4)
    if self.partition_occupancy * self.sub_tile_width <= 8:
      return 1.6 ** log2(self.partition_occupancy) * 1.33 ** log2(self.sub_tile_width) * self.tile_height * sequential_length
    else:
      return 4.7 * (self.partition_occupancy * self.sub_tile_width / 8) * self.tile_height * sequential_length

  def fitness(self):
    if self.model_type is ModelType.LSTM:
      sm_bandwidth = self.hidden_size * self.tile_height * 4
      warp_occupancy = ceil(self.num_threads / 128)

      self.mem_cost = round(sm_bandwidth * (1 + floor(self.num_SMs / (sms / 2))) / (fma_cost * l2_bus), 2)
      #print("Mem_cost:", mem_cost)
      self.sync_cost = 0 # ceil(self.num_SMs / 32) * 2
      #print("Sync_cost:", sync_cost)
      if warp_occupancy * self.sub_tile_width * self.tile_height < 12: ## Non-throughput limited 
        if self.reduction_width <= 16:
          self.reduction_cost = (log2(self.reduction_width) + 1) * 7 * 1.03 ** (warp_occupancy * self.sub_tile_width * self.tile_height)
        else:
          self.reduction_cost = log2(self.reduction_width) * 7 * 1.03 ** (warp_occupancy * self.sub_tile_width * self.tile_height)
      else: ## Throughput limited
        if self.reduction_width <= 16:
          self.reduction_cost = (log2(self.reduction_width) + 1) * 7 * 1.03 ** 8 * (warp_occupancy * self.sub_tile_width * self.tile_height / 12)
        else:
          self.reduction_cost = log2(self.reduction_width) * 7 * 1.03 ** 8 * (warp_occupancy * self.sub_tile_width * self.tile_height / 12)
      #print("Red_cost:", reduction_cost)
      self.mul_cost = round(self.fma_heuristic(), 2)
      #print("Mul_cost:", mul_cost)
      
      return self.mem_cost + self.sync_cost + self.reduction_cost + self.mul_cost

    elif self.model_type is ModelType.GRU:
      if self.sync is 2:
        sm_bandwidth = self.hidden_size * self.tile_height * 4 * 2
        warp_occupancy = ceil(self.num_threads / 128)

        self.mem_cost = round(sm_bandwidth * (1 + floor(self.num_SMs / (sms / 2))) / (fma_cost * l2_bus), 2)
        self.sync_cost = l2_lat * 2 * 2 # ceil(self.num_SMs / 32) * 2
        if warp_occupancy * self.sub_tile_width * self.tile_height < 12: ## Non-throughput limited 
          if self.reduction_width <= 16:
            self.reduction_cost = 2 * (log2(self.reduction_width) + 1) * 7 * 1.03 ** (warp_occupancy * self.sub_tile_width * self.tile_height)
          else:
            self.reduction_cost = 2 * log2(self.reduction_width) * 7 * 1.03 ** (warp_occupancy * self.sub_tile_width * self.tile_height)
        else: ## Throughput limited
          if self.reduction_width <= 16:
            self.reduction_cost = 2 * (log2(self.reduction_width) + 1) * 7 * 1.03 ** 8 * (warp_occupancy * self.sub_tile_width * self.tile_height / 12)
          else:
            self.reduction_cost = 2 * log2(self.reduction_width) * 7 * 1.03 ** 8 * (warp_occupancy * self.sub_tile_width * self.tile_height / 12)
        self.mul_cost = round(self.fma_heuristic() * 1.5, 2)
        return self.mem_cost + self.sync_cost + self.reduction_cost + self.mul_cost

      elif self.sync is 1:
        sm_bandwidth = self.hidden_size * self.tile_height * 4 + self.hidden_size * (self.hidden_size / self.tile_width) * self.tile_height * 4
        warp_occupancy = ceil(self.num_threads / 128)

        self.mem_cost = round(sm_bandwidth * (1 + floor(self.num_SMs / (sms / 2))) / (fma_cost * l2_bus), 2)
        self.sync_cost = 0
        if warp_occupancy * self.sub_tile_width * self.tile_height < 12: ## Non-throughput limited 
          if self.reduction_width <= 16:
            self.reduction_cost = 2 * (log2(self.reduction_width) + 1) * 7 * 1.03 ** (warp_occupancy * self.sub_tile_width * self.tile_height)
          else:
            self.reduction_cost = 2 * log2(self.reduction_width) * 7 * 1.03 ** (warp_occupancy * self.sub_tile_width * self.tile_height)
        else: ## Throughput limited
          if self.reduction_width <= 16:
            self.reduction_cost = 2 * (log2(self.reduction_width) + 1) * 7 * 1.03 ** 8 * (warp_occupancy * self.sub_tile_width * self.tile_height / 12)
          else:
            self.reduction_cost = 2 * log2(self.reduction_width) * 7 * 1.03 ** 8 * (warp_occupancy * self.sub_tile_width * self.tile_height / 12)
        self.mul_cost = round(self.fma_heuristic() + self.tile_width + self.hidden_size / self.tile_width, 2)
        return self.mem_cost + self.sync_cost + self.reduction_cost + self.mul_cost

  def to_csv(self):
    rep = str(self.tile_width) + ","
    rep += str(self.tile_height) + ","
    rep += str(self.num_work_groups) + "," 
    rep += str(self.reduction_width) + ","
    if self.model_type is ModelType.GRU:
      rep += str(self.sync) + ","
    rep += str(self.mem_cost) + ","
    rep += str(self.sync_cost) + ","
    rep += str(self.reduction_cost) + ","
    rep += str(self.mul_cost) + ","
    rep += str(self.cost) + ","
    rep += str(self.partition_occupancy * self.sub_tile_width < 16) + "\n"
    return rep

def string_to_model(string):
  if (string in string_model):
    return string_model[string]
  else:
    msg = string + " is not a valid model type"
    raise argparse.ArgumentTypeError(msg)


def main(model, input_size, hidden_size, batch_size, k):

  tile_widths = range(1, 65)
  tile_heights = range(1, batch_size + 1)
  
  tile_configurations = list(itertools.product(tile_widths, tile_heights))
  reduction_widths = [2 ** i for i in range(6)]

  configs = list()
  for x, y in tile_configurations:
    if batch_size % y is 0:
      num_gate_elements = x * model_gates[model]

      for i in range(1, num_gate_elements + 1):
        if num_gate_elements % i is 0:
          for r in reduction_widths:
            if model is ModelType.LSTM:
              configs.append(ModelConfig(model, hidden_size, batch_size, x, y, i, r))
            else:
              configs.append(ModelConfig(model, hidden_size, batch_size, x, y, i, r, sy=1))
              configs.append(ModelConfig(model, hidden_size, batch_size, x, y, i, r, sy=2))
  print(len(configs)) 
  configs = [x for x in configs if x.is_valid()]
  print(len(configs))
  configs.sort(key=lambda config: config.cost)
  
  with open("configs_" + str(hidden_size) + "_" + str(batch_size) + "_" + model_string[model] + ".csv", mode='w') as f:
    counter = 0
    for entry in configs:
      f.write(entry.to_csv())

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Use heuristic based analysis of \
                                                an RNN layer to determine a near optimal \
                                                configuration for instantiation')
  parser.add_argument('-m', '--model_type', default='LSTM', type=string_to_model, required=False,
                      help='The type of RNN layer for analysis')
  parser.add_argument('-i', '--input_size', default=256, type=int, required=False,
                      help='Length of input vector to layer')
  parser.add_argument('-s', '--hidden_size', default=256, type=int, required=False,
                      help='Length of hidden size/output of layer')
  parser.add_argument('-b', '--batch_size', default=1, type=int, required=False,
                      help='Size of batch to be computed simultaneously')
  parser.add_argument('-k', '--top_k', default=10, type=int, required=False,
                      help='How many candidate configurations to return')
  args = parser.parse_args()
  main(args.model_type, args.input_size, args.hidden_size, args.batch_size, args.top_k)
