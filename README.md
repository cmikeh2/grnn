# GRNN
## Framework Structure
### Inference Workflow
In GRNN, models are constructed individually by layer (See caveats in Design Problems) and then incorporated into a Model wrapper that holds collections of layers. Once the model has been defined with its given parameters, tiling parameters are passed to the model and the model is initialized. Layer initialization reorganizes the weight columns to improve locality of gate outputs, transposes the hidden state matrix, and sends these matrices to device memory. Model initialization allocates input, output, and intermediate buffers (this also feeds into design problems) as well as providing the known kernel parameters at this point.
At this point, inputs can be fed to the network. The model assumes a maximum sequence length (needs additional robustness) but otherwise is not constrained by the sequence of the provided batch. The batch elements are assumed to have the same sequence length.

### Design Problems
Do to the scope of the work, I made some design decisions pretty early in the process to prioritize single layer RNN models/benchmarking. As such, there is some overlap in the roles of the *Layer and *Model classes in GRNN. On the whole, *Layer classes are responsible for maintaining the parameters of the layer and performing the data reorganization of the trained parameters. The *Model classes tend to interact with the external interfaces of the system, currently including getting inputs and directly calling kernels. Since the model classes are currently responsible for execution, they are limited to single layers rather than stacked networks. The framework was built such that the amount of refactoring necessary is relatively small in order to enable these frameworks.  Since there I de-prioritized multilayer RNNs, there currently isn’t any code to validate that sequential layers 
A second limitation is currently cells of different types can not be in connected sequentially in a multilayer network. Since some incompatible information about LSTM and GRU kernel inference was include in the *Model classes, they needed to be separated. On a refactor, this would not be an issue.

## Kernels
The kernels for all cells/types follow the same broad three step process:
1. Buffer initialization - Initialize arrays in the register file and shared memory for the hidden/cell state and trained parameters.
2. Data Loading - Load weights and biases to register file, initialize shared cell/hidden states, and calculate offsets into the precompute array.
3. Recurrent Computation - varies based on cell structure.

## Performance Model
The performance model takes in model parameters, permutes the parameters to build the configuration space, prunes based on configuration feasibility, and then ranks based on the four-part performance model.

