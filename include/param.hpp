#ifndef _CNN_PARAM_
#define _CNN_PARAM_

struct conv_param
{
    int pad;
    int stride;
    int kernel_size;
    int in_channels;
    int out_channels;
    float *p_weight;
    float *p_bias;
};

struct fc_param
{
    int in_features;
    int out_features;
    float *p_weight;
    float *p_bias;
};

#endif