#include "cnn.hpp"
using namespace std;
using namespace cv;

inline float im2col_get_pixel(const float* im, int height, int width, int channels, int row, int col, int channel, int pad)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width) return 0;
	return im[col + width * (row + height * channel)];
}

void im2col_cpu(const float* data_im, int channels, int height, int width, int kernel_size, int stride, int pad, float* data_col)
{
	int c, h, w;
	int height_col = (height + 2 * pad - kernel_size) / stride + 1;
	int width_col = (width + 2 * pad - kernel_size) / stride + 1;
	int channels_col = channels * kernel_size * kernel_size;
	for (c = 0; c < channels_col; ++c) {
		int w_offset = c % kernel_size;
		int h_offset = (c / kernel_size) % kernel_size;
		int c_im = c / kernel_size / kernel_size;
		for (h = 0; h < height_col; ++h) {
			for (w = 0; w < width_col; ++w) {
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
			}
		}
	}
}

float* ConvBNReLU(float* neuron, const int height, const int width, const conv_param& param)
{
	int out_h = (height + 2 * param.pad - param.kernel_size) / param.stride + 1;
	int out_w = (width + 2 * param.pad - param.kernel_size) / param.stride + 1;
	const int M = param.out_channels;
	const int N = out_h * out_w;
	const int K = param.in_channels * param.kernel_size * param.kernel_size;
	float* matA = new float[M * K];
	float* matB = new float[K * N];
	float* matC = new float[M * N];
	im2col_cpu(neuron, param.in_channels, height, width, param.kernel_size, param.stride, param.pad, matB);
	for (int oc = 0; oc < param.out_channels; ++oc)
		for (int ks = 0; ks < param.in_channels * param.kernel_size * param.kernel_size; ++ks)
			matA[oc * K + ks] = param.p_weight[oc * K + ks];
	fast_sgemm(matA, matB, matC, M, K, N);
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			matC[i * N + j] += param.p_bias[i];
	for (int i = 0; i < M * N; ++i)
		if (matC[i] < 0)
			matC[i] = 0.0f;
	delete[] matA;
	delete[] matB;
	delete[] neuron;
	return matC;
}

float* MaxPoll2d(float* neuron, const int channels, const int height, const int width)
{
	float max = 0.0f;
	float* p1 = NULL, * p2 = NULL;
	const int out_h = height / 2, out_w = width / 2;
	float* out = new float[channels * out_h * out_w];
	for (int c = 0; c < channels; ++c)
	{
		p1 = neuron + c * height * width;
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

float* FullyCon(float* neuron, const fc_param& param)
{
	const int M = param.out_features;
	const int K = param.in_features;
	const int N = 1;
	float* out = new float[param.out_features * 1];
	fast_sgemm(param.p_weight, neuron, out, M, K, N);
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			out[i * N + j] += param.p_bias[i];
	delete[] neuron;
	return out;
}

void SoftMax(float* src, const int src_len)
{
	float max = src[0];
	float sum = 0.0f;
	for (int i = 1; i < src_len; i++)
		if (src[i] > max)
			max = src[i];
	for (int i = 0; i < src_len; i++)
		sum += expf(src[i] - max);
	for (int i = 0; i < src_len; i++)
		src[i] = expf(src[i] - max - log(sum));
}
