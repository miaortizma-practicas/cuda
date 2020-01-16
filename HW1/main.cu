#include <iostream>
#include <string>
#include <stdio.h>
#include "../utils.hpp"

size_t numRows();  //return # of rows in the image
size_t numCols();  //return # of cols in the image

void preProcess(uchar4 **h_rgbaImage, unsigned char **h_greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string& filename);

void postProcess(const std::string& output_file);

void your_rgba_to_greyscale(uchar4 * const d_rgbaImage, unsigned char* const d_greyImage, 
                            size_t numRows, size_t numCols);

#include "HW1.cu"

int main(int argc, char **argv) {
  uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  std::string input_file;
  std::string output_file;
  if (argc == 3) {
    input_file  = std::string(argv[1]);
    output_file = std::string(argv[2]);
  } else {
    std::cerr << "Usage: ./hw input_file output_file" << std::endl;
    exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);
  std::cout << "preprocess ok\n";
  const int N = 5;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      uchar4 rgba = h_rgbaImage[i * numCols() + j];
      std::cout << int(rgba.x) << "," << int(rgba.y) << "," << int(rgba.z) << "\t"; 
    }
    std::cout << "\n";
  }
  

  
  your_rgba_to_greyscale(d_rgbaImage, d_greyImage, numRows(), numCols());
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //check results and output the grey image
  postProcess(output_file);
  std::cout << "postprocess ok\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      uchar grey = h_greyImage[i * numCols() + j];
      std::cout << int(grey) << "\t"; 
    }
    std::cout << "\n";
  }
  std::cout << "Done!\n";
  return 0;
}