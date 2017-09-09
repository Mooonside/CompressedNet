// example.h
#ifndef BIN_APRX_H
#define BIN_APRX_H

template <typename Device, typename T>
struct BinAprx {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#endif