/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "../utils.hpp"

__global__ void myReduce(float* const logLum,
                         const int numRows, const int numCols,
                         const int step, const int op) 
{
    const int r = threadIdx.x + (blockIdx.x * blockDim.y);
    const int c = threadIdx.y + (blockIdx.y * blockDim.y);
    if (r >= numRows || c >= numCols) return;
    const int idx = r * numCols + c;
    const int p = 1 << (step + 1);
    const int o = idx + (1 << step);
    if (idx == 0) {
        printf("step: %d op: %d val: %f\n", step, op, logLum[idx]);
    }
    if (idx % p == 0 && o < numRows * numCols) {
        float a = logLum[idx], b = logLum[o];
        logLum[idx] = (op == 0) ? min(a, b) : max(a, b);
    }
    if (idx == 0) {
        printf("step: %d op: %d val: %f\n", step, op, logLum[idx]);
    }
}

__global__ void checkLuminance(const float* const logLuminance, 
                               const int numRows, const int numCols) {
    const int r = threadIdx.x + (blockIdx.x * blockDim.y);
    const int c = threadIdx.y + (blockIdx.y * blockDim.y);
    if (r >= numRows || c >= numCols) return;
    const int idx = r * numCols + c;
    if (idx <= 50) {
        printf("idx: %d val: %f\n", idx, logLuminance[idx]);
    }
} 

__global__ void checkHisto(unsigned int* const histo, const int numBins, 
                          const int numRows, const int numCols) {
    const int r = threadIdx.x + (blockIdx.x * blockDim.y);
    const int c = threadIdx.y + (blockIdx.y * blockDim.y);
    if (r >= numRows || c >= numCols) return;
    const int idx = r * numCols + c;
    if (idx < numBins && histo[idx] > 0) {
        printf("histo[%d]: %u total: %d\n", idx, histo[idx], numRows * numCols);
    }
} 


__global__ void addToBin(const float* const logLuminance, unsigned int* const histo,
                         const float logLumMin, const float logLumRange, 
                         const int numBins, 
                         const int numRows, const int numCols) {
    const int r = threadIdx.x + (blockIdx.x * blockDim.y);
    const int c = threadIdx.y + (blockIdx.y * blockDim.y);
    if (r >= numRows || c >= numCols) return;
    const int idx = r * numCols + c;
    unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((logLuminance[idx] - logLumMin) / logLumRange * numBins));
    atomicAdd(histo + bin, 1);
}

__global__ void sequentialCdfScan(unsigned int* const cdf, unsigned int* const histo,
                        const int numBins) 
{
    for (int i = 1; i < numBins; ++i) {
      cdf[i] = cdf[i - 1] + histo[i - 1];
    }
}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  const int N = 5, pixels = numRows * numCols;
  const int blockWidth = 1 << N;
  const dim3 blockSize(blockWidth, blockWidth, 1);
  const dim3 gridSize((numRows / blockWidth) + 1, (numCols / blockWidth) + 1, 1);
 
  checkLuminance<<<gridSize, blockSize>>>(d_logLuminance, numRows, numCols);
  
  float *h_min_logLum = new float[pixels], *h_max_logLum = new float[pixels],
        *d_min_logLum, *d_max_logLum;
  int size = sizeof(float) * pixels;
  checkCudaErrors(cudaMalloc((void **)&d_min_logLum, size));
  checkCudaErrors(cudaMalloc((void **)&d_max_logLum, size));
  checkCudaErrors(cudaMemcpy(d_min_logLum, d_logLuminance, size, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_max_logLum, d_logLuminance, size, cudaMemcpyDeviceToDevice));
  
  // step 1
  for (int i = 0; i < N; ++i) {
    myReduce<<<gridSize, blockSize>>>(d_min_logLum, numRows, numCols, i, 0);
    myReduce<<<gridSize, blockSize>>>(d_max_logLum, numRows, numCols, i, 1);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());     
  }
  
  checkCudaErrors(cudaMemcpy(h_min_logLum, d_min_logLum, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_max_logLum, d_max_logLum, size, cudaMemcpyDeviceToHost));
 
  for (int i = 0; i < blockWidth; ++i) {
    for (int j = 0; j < blockWidth; ++j) {
        int idx = (i * gridSize.x) * numCols + (j * gridSize.y);
        min_logLum = min(min_logLum, h_min_logLum[idx]);
        max_logLum = min(max_logLum, h_max_logLum[idx]);
    }
  }

  // step 2
  float logLumRange = max_logLum - min_logLum;
  
  // step 3
  unsigned int *d_histo;
  checkCudaErrors(cudaMalloc((void **)&d_histo, pixels * sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_histo, 0, pixels * sizeof(unsigned int)));
  printf("numBins: %zu\n", numBins);
  addToBin<<<gridSize, blockSize>>>(d_logLuminance, d_histo,
                                    min_logLum, logLumRange, numBins,
                                    numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());     
  checkHisto<<<gridSize, blockSize>>>(d_histo, numBins, numRows, numCols);
  // there is only 1024 bins, don't really need to parallelize or?
  sequentialCdfScan<<<1, 1>>>(d_cdf, d_histo, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());     
  
  checkCudaErrors(cudaFree(d_min_logLum));
  checkCudaErrors(cudaFree(d_max_logLum));
  checkCudaErrors(cudaFree(d_histo));
  
  printf("max: %f, min: %f\n", max_logLum, min_logLum);
}