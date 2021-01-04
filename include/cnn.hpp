#ifndef _CNN_
#define _CNN_
#include <omp.h>
#include <cmath>
#include <string>
#include <cstring>
#include <iostream>
#include "param.hpp"
#include <exception>
#include <immintrin.h>
#include <opencv2/opencv.hpp>
#define NUM_THREADS 12

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

using namespace std;
using namespace cv;

bool fast_sgemm(const float *A, const float *B, float *C, const size_t M, const size_t K, const size_t N);
void im2col_cpu(const float* data_im, int channels, int height, int width, int kernel_size, int stride, int pad, float* data_col);
float *ConvBNReLU(float *neuron, const int height, const int width, const conv_param &param);
float *MaxPoll2d(float *neuron, const int channels, const int height, const int width);
float *FullyCon(float *neuron, const fc_param &param);
void SoftMax(float* src, const int src_len);

class Img
{
private:
    int channels;
    int width;
    int height;
    float* data;
    float* score;
    friend void myAlloc(const string &full_path, Img &img);

public:
    Img();
    Img(const Img &img);
    Img(const string &full_path);
    ~Img();
    void scanner(const string &full_path);
    float *facedect();
};

#endif
