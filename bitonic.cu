// https://www.youtube.com/watch?v=j5ShDVuCyBg

#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include "utils.h"

void bitonic_CPU(double* A, long N) {
  for (long len = 1; len < N; len = len << 1) {
    for (long k = 0; k < N/(2*len); k++) {
      long offset = 2*k*len;
      for (long l = 0; l < len; l++) {
        if (A[offset+l] > A[offset+2*len-l-1]) {
          std::swap(A[offset+l], A[offset+2*len-l-1]);
        }
      }
    }
    for (long j = len/2; j >= 1; j=j>>1) {
      for (long k = 0; k < N/(2*j); k++) {
        long offset = (2*k+0)*j;
        for (long l = 0; l < j; l++) {
          if (A[offset+l] > A[offset+l+j]) std::swap(A[offset+l], A[offset+l+j]);
        }
      }
    }
  }
}


void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

__device__ void swap(double& a, double& b) {
  double c = a;
  a=b;
  b=c;
}

__global__ void bitonic_merge_kernel0(double* A, long N, long len) {
  long idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
  __shared__ double smem[2048];

  if (idx < N) smem[threadIdx.x] = A[idx];
  if (idx + blockDim.x < N) smem[threadIdx.x + blockDim.x] = A[idx + blockDim.x];
  __syncthreads();

  long k = threadIdx.x / len;
  long offset = 2 * k * len;
  long l = threadIdx.x - k * len;
  if (smem[offset+l] > smem[offset+2*len-l-1]) swap(smem[offset+l], smem[offset+2*len-l-1]);
  __syncthreads();

  for (long j = len/2; j >= 1; j=j>>1) {
    long k = threadIdx.x / j;
    long offset = 2 * k * j;
    long l = threadIdx.x - k * j;
    if (smem[offset+l] > smem[offset+l+j]) swap(smem[offset+l], smem[offset+l+j]);
    __syncthreads();
  }

  A[idx] = smem[threadIdx.x];
  A[idx + blockDim.x] = smem[threadIdx.x + blockDim.x];
}

__global__ void bitonic_compare_swap_kernel0(double* A, long N, long len) {
  long idx = threadIdx.x + blockIdx.x * blockDim.x;

  long k = idx / len;
  long offset = 2 * k * len;
  long l = idx - k * len;
  if (A[offset+l] > A[offset+2*len-l-1]) swap(A[offset+l], A[offset+2*len-l-1]);
}

__global__ void bitonic_merge_kernel1(double* A, long N, long len) {
  long idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
  __shared__ double smem[2048];

  if (idx < N) smem[threadIdx.x] = A[idx];
  if (idx + blockDim.x < N) smem[threadIdx.x + blockDim.x] = A[idx + blockDim.x];
  __syncthreads();

  for (long j = len; j >= 1; j=j>>1) {
    long k = threadIdx.x / j;
    long offset = 2 * k * j;
    long l = threadIdx.x - k * j;
    if (smem[offset+l] > smem[offset+l+j]) swap(smem[offset+l], smem[offset+l+j]);
    __syncthreads();
  }

  A[idx] = smem[threadIdx.x];
  A[idx + blockDim.x] = smem[threadIdx.x + blockDim.x];
}

__global__ void bitonic_compare_swap_kernel1(double* A, long N, long len) {
  long idx = threadIdx.x + blockIdx.x * blockDim.x;

  long k = idx / len;
  long offset = 2 * k * len;
  long l = idx - k * len;
  if (A[offset+l] > A[offset+l+len]) swap(A[offset+l], A[offset+l+len]);
}

void bitonic_merge_GPU(double* A, long N, long len) {
  long gridDim = (N+2048-1)/2048;
  long blockDim = std::min<long>(1024, N/2);
  if (len <= 1024) {
    bitonic_merge_kernel0<<<gridDim, blockDim>>>(A, N, len);
  } else {
    bitonic_compare_swap_kernel0<<<gridDim, blockDim>>>(A, N, len);
    for (long j = len/2; j >= 1024; j=j>>1) {
      bitonic_compare_swap_kernel1<<<gridDim, blockDim>>>(A, N, j);
    }
    bitonic_merge_kernel1<<<gridDim, blockDim>>>(A, N, std::min<long>(1024, len/2));
  }
}

void bitonic_GPU(double* A, long N) {
  for (long len = 1; len < N; len = len << 1) {
    bitonic_merge_GPU(A, N, len);
    cudaDeviceSynchronize();
  }
}


int main() {
  long N = (1u<<25);
  double* A0 = (double*) malloc(N*sizeof(double));
  double* A1 = (double*) malloc(N*sizeof(double));
  double* A2 = (double*) malloc(N*sizeof(double));
  for (long i = 0; i < N; i++) {
    A0[i] = drand48();
    A1[i] = A0[i];
    A2[i] = A0[i];
  }

  Timer t;
  t.tic();
  std::sort(A0, A0 + N);
  double tt = t.toc();
  printf("CPU throughput (std::sort) = %.2f million-keys/s\n", N/tt*1e-6);

  t.tic();
  bitonic_CPU(A1, N);
  tt = t.toc();
  printf("CPU throughput (bitonic) = %.2f million-keys/s\n", N/tt*1e-6);

  double *A2_d;
  cudaMalloc(&A2_d, N*sizeof(double));
  cudaMemcpy(A2_d, A2, N*sizeof(double), cudaMemcpyHostToDevice);

  t.tic();
  bitonic_GPU(A2_d, N);
  cudaDeviceSynchronize();
  tt = t.toc();
  printf("GPU throughput (bitonic) = %.2f million-keys/s\n", N/tt*1e-6);
  cudaMemcpy(A2, A2_d, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  double err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, fabs(A0[i] - A1[i]));
  for (long i = 0; i < N; i++) err = std::max(err, fabs(A0[i] - A2[i]));
  printf("Error = %f\n", err);

  free(A0);
  free(A1);
  free(A2);
  cudaFree(A2_d);

  return 0;
}
