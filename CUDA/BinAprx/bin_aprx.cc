// example.cc
#define EIGEN_USE_THREADS
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/register_types.h"
#include "bin_aprx.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("BinAprx")
	//inferred attrs are given Capitalized or CamelCase names.
    .Attr("T: {int32,float} = DT_INT32")
    .Input("in: T")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    ; 

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct BinAprx<CPUDevice, T> {
  //override () operator
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    // Set all but the first element of the output tensor to 0.
    float alpha = 0;

    for (int i = 0; i < size; i++){  
      out[i] = in[i] < 0 ? (-1) : (1);
      alpha += out[i] * in[i];
    }
    
    alpha = alpha / (float)size ;
    for (int i = 0; i < size; i++){  
      out[i] *= alpha;
    }
  }
};

//OpKernel definition.
//template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class BinAprxOp : public OpKernel {
 public:
  explicit BinAprxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    // Create an output tensor
    Tensor* output_tensor = NULL;
    // Allocate memory for output tensor
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    BinAprx<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BinAprx").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BinAprxOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);



#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BinAprx").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BinAprxOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);