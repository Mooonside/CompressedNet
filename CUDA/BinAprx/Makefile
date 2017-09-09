TF_INC='/usr/local/lib/python2.7/dist-packages/tensorflow/include'

all:
	nvcc -std=c++11 -c -o bin_aprx.cu.o bin_aprx.cu.cc -I $(TF_INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr
	g++ -std=c++11 -shared -o bin_aprx.so bin_aprx.cc bin_aprx.cu.o -I $(TF_INC) -fPIC -lcudart -D_GLIBCXX_USE_CXX11_ABI=0 -L /usr/local/cuda-8.0/lib64/

clean:
	rm -f bin_aprx.cu.o
	rm -f bin_aprx.so