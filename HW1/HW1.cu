#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <string>

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename) {
  checkCudaErrors(cudaFree(0));

  cv::Mat image;
  image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);


  int cols = numCols();
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);

  imageGrey.create(image.rows, image.cols, CV_8UC1);

  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));

  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); 

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
  const int numPixels = numRows() * numCols();
  checkCudaErrors(cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
  //output the image
  cv::imwrite(output_file.c_str(), imageGrey);

  //cleanup
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}

__global__ void 
rgba_to_greyscale(const uchar4* const rgbaImage, unsigned char* const greyImage,
                       int numRows, int numCols)
{
  const int r = threadIdx.x + blockIdx.x * blockDim.x; 
  const int c = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = r * numCols + c;
  if (r < numRows && c < numCols) {
    int idx = r * numCols + c;
    uchar4 rgba = rgbaImage[idx];
    unsigned char grey = (unsigned char)(0.299f*rgba.x+ 0.587f*rgba.y + 0.114f*rgba.z);
    greyImage[idx] = grey;
  } 
}

void your_rgba_to_greyscale(uchar4 * const d_rgbaImage, unsigned char* const d_greyImage, 
                            size_t numRows, size_t numCols) 
{
  const int blockWidth = 32;
  const dim3 blockSize(blockWidth, blockWidth, 1);
  const dim3 gridSize(floor(numCols/blockWidth)+1,floor(numRows/blockWidth)+1, 1);
  fprintf(stdout, "numRows: %zu, numCols: %zu\n", numRows, numCols);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}