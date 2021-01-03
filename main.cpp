inline float im2col_get_pixel(const float *im, int height, int width, int channels, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width)
        return 0;
    return im[col + width * (row + height * channel)];
}

void im2col_cpu(const float *data_im, int channels, int height, int width, int kernel_size, int stride, int pad, float *data_col)
{
    int height_col = (height + 2 * pad - kernel_size) / stride + 1;
    int width_col = (width + 2 * pad - kernel_size) / stride + 1;
    int channels_col = channels * kernel_size * kernel_size;
#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for
#endif
    for (int c = 0; c < channels_col; ++c)
    {
        int w_offset = c % kernel_size;
        int h_offset = (c / kernel_size) % kernel_size;
        int c_im = c / kernel_size / kernel_size;
        for (int h = 0; h < height_col; ++h)
        {
            for (int w = 0; w < width_col; ++w)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
}

float *ConvBNReLU(float *neuron, const int height, const int width, const conv_param &parma)
{
    int out_h = (height + 2 * parma.pad - parma.kernel_size) / parma.stride + 1;
    int out_w = (width + 2 * parma.pad - parma.kernel_size) / parma.stride + 1;
    int M = parma.out_channels;
    int N = out_h * out_w;
    int K = parma.in_channels * parma.kernel_size * parma.kernel_size;
    float *matA = new float[M * K];
    float *matB = new float[K * N];
    float *matC = new float[M * N];
    im2col_cpu(neuron, parma.in_channels, height, width, parma.kernel_size, parma.stride, parma.pad, matB);
#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for
#endif
    for (int oc = 0; oc < parma.out_channels; ++oc)
        for (int ks = 0; ks < parma.in_channels * parma.kernel_size * parma.kernel_size; ++ks)
            matA[oc * K + ks] = parma.p_weight[oc * K + ks];
    fast_sgemm(matA, matB, matC, M, K, N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N;++j)
            matC[i * N + j] += parma.p_bias[i];
    for (int i = 0; i < M * N; ++i)
        if (matC[i] < 0)
            matC[i] = 0;
    delete[] matA;
    delete[] matB;
    delete[] neuron;
    return matC;
}

float* MaxPoll2d(float* neuron, const int channels, const int height, const int width)
{
    float max;
    float* p1, * p2;
    int out_h = height / 2, out_w = width / 2;
    float* out = new float[channels * out_h * out_w];
    for (int c = 0; c < channels; ++c)
    {
        p1 = const_cast<float*>(neuron + c * height * width);
        p2 = p1 + width;
        for (int h = 0; h < height / 2; ++h)
        {
            for (int w = 0; w < width / 2; ++w)
            {
                max = MAX(*p1, *p2);
                ++p1;
                ++p2;
                max = MAX(max, *p1);
                max = MAX(max, *p2);
                out[c * out_h * out_w + h * out_w + w] = max;
                ++p1;
                ++p2;
            }
            p1 += width;
            p2 += width;
        }
    }
    delete[] neuron;
    return out;
}
