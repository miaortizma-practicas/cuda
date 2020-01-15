#include <cuda.h>
#include <cuda_runtime.h>

inline cudaError_t checkCudaErrors(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime error: %s\n", cudaGetErrorString(err));
  }
  return err;
}