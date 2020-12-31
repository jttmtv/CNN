#ifndef _FAST_SGEMM_
#define _FAST_SGEMM_
#include <cmath>
#include <cstring>
#include <exception>
#include <immintrin.h>
#include <iostream>
#include <omp.h>

bool fast_sgemm(const float *A, const float *B, float *C, const size_t M, const size_t K, const size_t N);

#endif