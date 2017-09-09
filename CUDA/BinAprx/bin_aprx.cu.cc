  #ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "bin_aprx.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

// Define the CUDA kernel.
template <typename T>
__global__ void reduce_blk(const int size, const T* in, T* out) {
    T *sdata = SharedMemory<T>();
    // load shared mem
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = (i < size) ? in[i] : 0;
    sdata[tid] = sdata[tid] < 0 ? (-sdata[tid]) : sdata[tid]; 

    __syncthreads();

    // do reduction in shared mem
    for (int s=blockDim.x/2; s>0; s>>=1)
    {
        // if (tid < s)
        if (tid < s && ( tid + s < blockDim.x))
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) out[blockIdx.x] = sdata[0];
};


template <class T>
T reduce_recursive(const T *inputs, int n){
  int BlockSize = 256;
  int BlockNum = (float)(n + BlockSize - 1) / BlockSize;
  // std::cout << BlockSize << " " << BlockNum << std::endl;
  T *outputs;
  cudaMallocManaged(&outputs, BlockNum*sizeof(T));
  reduce_blk<T>
    <<<BlockNum, BlockSize,BlockSize*sizeof(T)>>>
      (n,inputs, outputs);
  cudaDeviceSynchronize();
  // return 0;

  if(BlockNum >= 256){
    T sum = reduce_recursive(outputs,BlockNum);
    cudaFree(outputs);
    return sum;
  }
  else{
    T sum = outputs[0];
    for (int i = 1; i< BlockNum; i++){
      sum += outputs[i];
    }
    cudaFree(outputs);
    return sum;
  }
};

template <typename T>
__global__ void alpha_sign(const int size, const T* in, T* out,T alpha) {
    // unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i < size){
      out[i] = in[i] < 0 ? (-1) : 1;
      out[i] *= alpha;      
    }
};

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct BinAprx<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* in, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int BlockSize = 256;
    int BlockNum = (float)(size + BlockSize - 1) / BlockSize;

    T alpha = reduce_recursive<T>(in, size);
    alpha /= size;
    alpha_sign<T>
      <<<BlockNum,BlockSize>>>
        (size, in, out, alpha);

  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct BinAprx<GPUDevice, float>;
template struct BinAprx<GPUDevice, int32>;

#endif  // GOOGLE_CUDA