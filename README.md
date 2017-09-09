# CompressedNet
Differnent approaches to compress and accelerate CNNs

## Stucture:
-./Utils stores different conv|dense layers used in compressed nets.
-./CUDA stores the cuda kernel for custom ops/

## Models:
- Pruned_LeNet : LeNet that prunes unnecessary connections.
- Shared_LeNet: LeNet that shares weight in Conv|Dense Kernels.
- Bin_LeNet: LeNet that has binary weights.
- Xnor_LeNet: LeNet that has binary weights and activations.
