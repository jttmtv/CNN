#include <iostream>
using namespace std;

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

float *MaxPoll2d(float *neuron, const int channels, const int height, const int width)
{
    float max;
    float *p1, *p2;
    int out_h = height / 2, out_w = width / 2;
    float *out = new float[channels * out_h * out_w];
    for (int c = 0; c < channels; ++c)
    {
        p1 = const_cast<float *>(neuron + c * height * width);
        p2 = p1 + width;
        for (int h = 0; h < height / 2; ++h)
        {
            for (int w = 0; w < width / 2; ++w)
            {
                max = MAX(*p1, *p2);
                max = MAX(max, *(++p1));
                max = MAX(max, *(++p2));
                out[c * out_h * out_w + h * out_w + w] = max;
                ++p1;
                ++p2;
            }
            p1 += width;
            p2 += width;
        }
    }
    //delete[] neuron;
    return out;
}

int main()
{
    float *a = new float[4 * 4 * 3];
    float *b;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                a[i * 4 * 4 + j * 4 + k] = k;
    b = MaxPoll2d(a, 3, 4, 4);
    for (int i = 0; i < 4 * 4 * 3; ++i)
        cout << a[i] << " ";
    printf("\n");
    for (int i = 0; i < 2 * 2 * 3; ++i)
        cout << b[i] << " ";
    printf("\n");
}
