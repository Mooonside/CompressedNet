# CompressedNet
Differnent approaches to compress and accelerate CNNs

## References:
- Deep Compression-Compressing Deep Neural Networks with Pruning%2c Trained Quantization and Huffman Coding.
- XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.

## Stucture:
- ./Utils stores different conv|dense layers used in compressed nets.
- ./CUDA stores the cuda kernel for custom ops/

## Models:
- Pruned_LeNet : LeNet that prunes unnecessary connections.
- Shared_LeNet: LeNet that shares weight in Conv|Dense Kernels.
- Bin_LeNet: LeNet that has binary weights.
- Xnor_LeNet: LeNet that has binary weights and activations.

## Coming Soon:
- XNOR Convolution GPU Kernel.
