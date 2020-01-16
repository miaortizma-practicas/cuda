#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <string>

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }

void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                float **h_filter, int *filterWidth,
                const std::string &filename) {

  checkCudaErrors(cudaFree(0));

  cv::Mat image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageInputRGBA, cv::COLOR_BGR2RGBA);
  imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);
  if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
  *h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  printf("image dimensions: %zu %zu\n", numRows(), numCols());
  checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_inputImageRGBA__  = *d_inputImageRGBA;
  d_outputImageRGBA__ = *d_outputImageRGBA;

  //now create the filter that they will use
  const int blurKernelWidth = 9;
  const float blurKernelSigma = 2.;

  *filterWidth = blurKernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  h_filter__ = *h_filter;

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;
  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }

  //blurred
  checkCudaErrors(cudaMalloc(d_redBlurred,    sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels));
}

void postProcess(const std::string& output_file) {
  const int numPixels = numRows() * numCols();
  //copy the output back to the host
  checkCudaErrors(cudaMemcpy(imageOutputRGBA.ptr<unsigned char>(0), d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

  cv::Mat imageOutputBGR;
  cv::cvtColor(imageOutputRGBA, imageOutputBGR, cv::COLOR_RGBA2BGR);
  //output the image
  cv::imwrite(output_file.c_str(), imageOutputBGR);

  //cleanup
  cudaFree(d_inputImageRGBA__);
  cudaFree(d_outputImageRGBA__);
  delete[] h_filter__;
}

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  const int r = threadIdx.x + blockIdx.x * blockDim.x; 
  const int c = threadIdx.y + blockIdx.y * blockDim.y;
  if ( r >= numCols || c >= numRows ) {
    return;
  }
  const int idx = r * numCols + c;
  float avg = 0;
  for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
    for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
      int image_r = min(max(r + filter_r, 0), static_cast<int>(numRows - 1));
      int image_c = min(max(c + filter_c, 0), static_cast<int>(numCols - 1));
      float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
      float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];
      avg += image_value * filter_value;   
    }
  }
 
  outputChannel[idx] = avg;
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  const int r = threadIdx.x + blockIdx.x * blockDim.x; 
  const int c = threadIdx.y + blockIdx.y * blockDim.y;
  if ( r >= numCols || c >= numRows ) {
    return;
  }
  int idx = r * numCols + c;
  uchar4 rgba = inputImageRGBA[idx];
  redChannel[idx] = rgba.x;
  greenChannel[idx] = rgba.y;
  blueChannel[idx] = rgba.z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.x * numCols + thread_2D_pos.y;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numRows || thread_2D_pos.y >= numCols)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_filter,  sizeof(float) * filterWidth * filterWidth));

  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  const int blockWidth = 32;
  const dim3 blockSize(blockWidth, blockWidth, 1);

  const dim3 gridSize(floor(numRows / blockWidth) + 1, floor(numCols / blockWidth) + 1, 1);

  // Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
 
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


//Free all the memory that we allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}