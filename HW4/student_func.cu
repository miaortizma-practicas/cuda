//Udacity HW 4
//Radix Sorting

#include "../utils.hpp"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void addToBin(uint* const inputVals, 
                         const uint mask, const uint bit,
                         uint* const histo, const int numElems) 
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= numElems) return;
    uint bin = (inputVals[idx] & mask) >> bit;
    atomicAdd(histo + bin, 1);
}

__global__ void seqExclusiveScan(uint* const scan, uint* const histo,
                                 const int numBins) 
{
  scan[0] = 0;
  for (int i = 1; i < numBins; ++i) {
    scan[i] = scan[i - 1] + histo[i - 1];
  }
  
  for (int i = 0; i < numBins; ++i) {
      //printf("histo[%d] = %u\n", i, histo[i]);
      //printf("histoScan[%d] = %u\n", i, scan[i]);
  }
  //printf("sum: %u\n", histo[0] + histo[1]);
}

__global__ void calculatePredicate(uint* const inputVals, uint* const scan,
                                  const uint mask, const uint bit, 
                                  const uint bin, const int numElems) 
{
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= numElems) return;
  uint valBin = (inputVals[idx] & mask) >> bit;
  if (valBin == bin) {
      scan[idx] = 1;
  } else {
      scan[idx] = 0;
  }
}

// hills steele
__global__ void myScan(uint* const scan, uint* const scanbuffer, 
                       const int numElems, const int step) 
{
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= numElems) return;
  scanbuffer[idx] = scan[idx];
  const int o = idx - (1 << step);
  if (o >= 0) {
    scanbuffer[idx] += scan[o];
  }
  if (idx == 0) {
      //printf("scan step: %d\n", step);
  }
}

__global__ void checkScan(const uint* const inputVals, const uint * const scan0, const uint * const scan1,
                          const uint mask, const uint bit) {
  for (int i = 0; i < 50; ++i) {
      uint bin = (inputVals[i] & mask) >> bit;
      //printf("idx: %d bin: %u scan0: %u scan1: %u \n", i, bin, scan0[i], scan1[i]);
  }
}

__global__ void setPos(uint* const pos, const uint* const inputVals,
                       const uint* const scan0, const uint* const scan1,
                       const uint* const histoScan,
                       const uint mask, const uint bit,
                       const int numElems) 
{
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= numElems) return;
  uint bin = (inputVals[idx] & mask) >> bit;
  if (bin == 0) {
    pos[idx] = scan0[idx] - 1 + histoScan[0];
    if (pos[idx] >= numElems) {
      //printf("Problems at: %d pos: %u\t scan0[%d]: %u histoScan[0]: %u\n", idx, pos[idx], idx, scan0[idx], histoScan[0]);
    }
  } else {
    pos[idx] = scan1[idx] - 1 + histoScan[1];
    if (pos[idx] >= numElems) {
      //printf("Problems at: %d pos: %u\t scan1[%d]: %u histoScan[1]: %u\n", idx, pos[idx], idx, scan1[idx], histoScan[1]);
    }
  }
    
}

__global__ void scatter(uint* const inputVals, uint* const inputPos,
                        uint* const outputVals, uint* const outputPos,
                        const uint* const pos, const int numElems) 
{
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= numElems) return;
  const uint p = pos[idx];
  outputVals[p] = inputVals[idx];
  outputPos[p] = inputPos[idx];
}

void your_sort(unsigned int* d_inputVals,
               unsigned int* d_inputPos,
               unsigned int* d_outputVals,
               unsigned int* d_outputPos,
               const size_t numElems)
{
  printf("%zu\n", numElems);
  const int blockWidth = 32;
  const int blockSize = blockWidth * blockWidth;
  const int gridSize = numElems / blockSize + 1;
 
  const int numBits = 1;
  const int numBins = 1 << numBits;
 
  uint *d_0scan, *d_1scan,
      *d_pos,
      *d_0scanbuffer, *d_1scanbuffer,
      *d_histo, *d_histoScan;

  const int size = sizeof(unsigned int) * numBins;
  const int arraySize = sizeof(unsigned int) * numElems;
 
  checkCudaErrors(cudaMalloc((void **)&d_histo, size));
  checkCudaErrors(cudaMalloc((void **)&d_histoScan, size));
  checkCudaErrors(cudaMalloc((void **)&d_pos, arraySize));
  checkCudaErrors(cudaMalloc((void **)&d_0scan, arraySize));
  checkCudaErrors(cudaMalloc((void **)&d_1scan, arraySize));
  checkCudaErrors(cudaMalloc((void **)&d_0scanbuffer, arraySize));
  checkCudaErrors(cudaMalloc((void **)&d_1scanbuffer, arraySize));
  
  for (uint bit = 0; bit < 8 * sizeof(uint); bit += numBits) {
    uint mask = (numBins - 1) << bit;

    checkCudaErrors(cudaMemset(d_histo, 0, size));

    addToBin<<<gridSize, blockSize>>>(d_inputVals, mask, bit, d_histo, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    seqExclusiveScan<<<1, 1>>>(d_histoScan, d_histo, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //checkCudaErrors(cudaMemset(d_0scan, 0, arraySize));
    //checkCudaErrors(cudaMemset(d_1scan, 0, arraySize));

    calculatePredicate<<<gridSize, blockSize>>>(d_inputVals, d_0scan, 
                                                mask, bit, 0,
                                                numElems);
    calculatePredicate<<<gridSize, blockSize>>>(d_inputVals, d_1scan, 
                                                mask, bit, 1,
                                                numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    uint p = 0;
    while ((1 << p) < numElems) {
      myScan<<<gridSize, blockSize>>>(d_0scan, d_0scanbuffer, numElems, p);
      myScan<<<gridSize, blockSize>>>(d_1scan, d_1scanbuffer, numElems, p);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      std::swap(d_0scan, d_0scanbuffer);
      std::swap(d_1scan, d_1scanbuffer);
      ++p;
    }
    
    checkScan<<<1, 1>>>(d_inputVals, d_0scan, d_1scan, mask, bit);
    //Gather everything into the correct location
    //need to move vals and positions
    setPos<<<gridSize, blockSize>>>(d_pos, d_inputVals, 
                                    d_0scan, d_1scan, d_histoScan, 
                                    mask, bit, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    //printf("to scatter! step: %d\n", bit);
    scatter<<<gridSize, blockSize>>>(d_inputVals, d_inputPos, 
                                     d_outputVals, d_outputPos,
                                     d_pos, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    

    //swap the buffers (pointers only)
    std::swap(d_inputVals, d_outputVals);
    std::swap(d_inputPos, d_outputPos);
  }
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, arraySize, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, arraySize, cudaMemcpyDeviceToDevice));
}