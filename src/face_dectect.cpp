#include "cnn.hpp"
#include "weight.hpp"
using namespace std;
using namespace cv;

void myAlloc(const cv::String& full_path, Img& img)
{
    int start_pos;
    uchar* uc_pixel;
    img.height = 128;
    img.width = 128;
    img.channels = 3;
    Mat cvmat = imread(full_path, IMREAD_COLOR);
    if (!img.data){
        img.data = new float[128 * 128 * 3];
        img.score = new float[1 * 2];
    }
    for (int h = 0; h < cvmat.rows; ++h)
    {
        uc_pixel = cvmat.data + h * cvmat.step;
        for (int w = 0; w < cvmat.cols; ++w)
        {
            start_pos = 128 * h + w;
            img.data[start_pos] = uc_pixel[2] / 255.0f;                 //R
            img.data[128 * 128 + start_pos] = uc_pixel[1] / 255.0f;     //G
            img.data[128 * 128 * 2 + start_pos] = uc_pixel[0] / 255.0f; //B
            uc_pixel += 3;
        }
    }
}

Img::Img()
{
    height = 0;
    width = 0;
    channels = 0;
    score = NULL;
    data = NULL;
}

Img::Img(const Img& img)
{
    channels = img.channels;
    height = img.height;
    width = img.width;
    score = new float[1 * 2];
    data = new float[channels * height * width];
    memcpy(data, img.data, sizeof(float) * channels * height * width);
}

Img::Img(const cv::String& full_path)
{
    data = NULL;
    score = NULL;
    myAlloc(full_path, *this);
}

Img::~Img()
{
    delete[] score;
    delete[] data;
}

void Img::scanner(const cv::String& full_path)
{
    myAlloc(full_path, *this);
}

float* Img::facedect()
{
    float* neuron = NULL;
    float* dst = new float[channels * height * width];
    memcpy(dst, data, sizeof(float) * channels * height * width);
    neuron = ConvBNReLU(dst, height, width, conv_params[0]);
    neuron = MaxPoll2d(neuron, conv_params[0].out_channels, 64, 64);
    neuron = ConvBNReLU(neuron, 32, 32, conv_params[1]);
    neuron = MaxPoll2d(neuron, conv_params[1].out_channels, 30, 30);
    neuron = ConvBNReLU(neuron, 15, 15, conv_params[2]);
    FullyCon(neuron, score, fc_params[0]);
    SoftMax(score, 2);
    return score;
}
