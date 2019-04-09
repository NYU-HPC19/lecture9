#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include "utils.h"

void scan(double* scan, const double* a, long N){ // inclusive scan
  double sum = 0;
  for (long i = 0; i < N; i++) {
    scan[i] = sum;
    sum += a[i];
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

__global__ void scan_kernel(double* scan, double* reduc, const double* a, long N){
  long idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  __shared__ double reduc10[1024];
  __shared__ double reduc9[512];
  __shared__ double reduc8[256];
  __shared__ double reduc7[128];
  __shared__ double reduc6[64];
  __shared__ double reduc5[32];
  __shared__ double reduc4[32];
  __shared__ double reduc3[32];
  __shared__ double reduc2[32];
  __shared__ double reduc1[32];
  __shared__ double reduc0[32];

  __shared__ double scan1[32];
  __shared__ double scan2[32];
  __shared__ double scan3[32];
  __shared__ double scan4[32];
  __shared__ double scan5[32];
  __shared__ double scan6[64];
  __shared__ double scan7[128];
  __shared__ double scan8[256];
  __shared__ double scan9[512];
  __shared__ double scan10[1024];

  reduc10[threadIdx.x] = 0;
  if (idx < N) reduc10[threadIdx.x] = a[idx];

  { // build reduction tree
    __syncthreads();
    if (threadIdx.x < 512) reduc9[threadIdx.x] = reduc10[threadIdx.x*2]+ reduc10[threadIdx.x*2+1];
    __syncthreads();
    if (threadIdx.x < 256) reduc8[threadIdx.x] = reduc9[threadIdx.x*2]+ reduc9[threadIdx.x*2+1];
    __syncthreads();
    if (threadIdx.x < 128) reduc7[threadIdx.x] = reduc8[threadIdx.x*2]+ reduc8[threadIdx.x*2+1];
    __syncthreads();
    if (threadIdx.x <  64) reduc6[threadIdx.x] = reduc7[threadIdx.x*2]+ reduc7[threadIdx.x*2+1];
    __syncthreads();
    if (threadIdx.x <  32) {
      reduc5[threadIdx.x] = reduc6[threadIdx.x*2]+ reduc6[threadIdx.x*2+1];
      __syncwarp();
      reduc4[threadIdx.x] = reduc5[threadIdx.x*2]+ reduc5[threadIdx.x*2+1];
      __syncwarp();
      reduc3[threadIdx.x] = reduc4[threadIdx.x*2]+ reduc4[threadIdx.x*2+1];
      __syncwarp();
      reduc2[threadIdx.x] = reduc3[threadIdx.x*2]+ reduc3[threadIdx.x*2+1];
      __syncwarp();
      reduc1[threadIdx.x] = reduc2[threadIdx.x*2]+ reduc2[threadIdx.x*2+1];
      __syncwarp();
      reduc0[threadIdx.x] = reduc1[threadIdx.x*2]+ reduc1[threadIdx.x*2+1];
      __syncwarp();
    }
  }
  { // build scan tree
    if (threadIdx.x <   32) {
      scan1[threadIdx.x*2+0] = reduc0[threadIdx.x] - reduc1[threadIdx.x*2+1];
      scan1[threadIdx.x*2+1] = reduc0[threadIdx.x];
      __syncwarp();
      scan2[threadIdx.x*2+0] = scan1[threadIdx.x] - reduc2[threadIdx.x*2+1];
      scan2[threadIdx.x*2+1] = scan1[threadIdx.x];
      __syncwarp();
      scan3[threadIdx.x*2+0] = scan2[threadIdx.x] - reduc3[threadIdx.x*2+1];
      scan3[threadIdx.x*2+1] = scan2[threadIdx.x];
      __syncwarp();
      scan4[threadIdx.x*2+0] = scan3[threadIdx.x] - reduc4[threadIdx.x*2+1];
      scan4[threadIdx.x*2+1] = scan3[threadIdx.x];
      __syncwarp();
      scan5[threadIdx.x*2+0] = scan4[threadIdx.x] - reduc5[threadIdx.x*2+1];
      scan5[threadIdx.x*2+1] = scan4[threadIdx.x];
      __syncwarp();
      scan6[threadIdx.x*2+0] = scan5[threadIdx.x] - reduc6[threadIdx.x*2+1];
      scan6[threadIdx.x*2+1] = scan5[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x <   64) {
      scan7[threadIdx.x*2+0] = scan6[threadIdx.x] - reduc7[threadIdx.x*2+1];
      scan7[threadIdx.x*2+1] = scan6[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x <  128) {
      scan8[threadIdx.x*2+0] = scan7[threadIdx.x] - reduc8[threadIdx.x*2+1];
      scan8[threadIdx.x*2+1] = scan7[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x <  256) {
      scan9[threadIdx.x*2+0] = scan8[threadIdx.x] - reduc9[threadIdx.x*2+1];
      scan9[threadIdx.x*2+1] = scan8[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x <  512) {
      scan10[threadIdx.x*2+0] = scan9[threadIdx.x] - reduc10[threadIdx.x*2+1];
      scan10[threadIdx.x*2+1] = scan9[threadIdx.x];
    }
    __syncthreads();
  }

  if (idx < N) scan[idx] = scan10[threadIdx.x]-reduc10[threadIdx.x];
  if (threadIdx.x == 0) reduc[blockIdx.x] = reduc0[0];
}

__global__ void add_offset_kernel(double* scan, const double* offset, long N){
  long idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < N) scan[idx] += offset[blockIdx.x];
}

void gpu_scan(double* scan, double* a, long N) {
  double *a_, *scan_;
  long Nb = (N + 1024 - 1) / 1024;
  cudaMalloc(&a_, Nb*sizeof(double));
  cudaMalloc(&scan_, Nb*sizeof(double));

  scan_kernel<<<Nb,1024>>>(scan, a_, a, N);
  if (Nb > 1) {
    gpu_scan(scan_, a_, Nb);
    add_offset_kernel<<<Nb, 1024>>>(scan, scan_, N);
  }

  cudaFree(a_);
  cudaFree(scan_);
}

int main() {
  long N = (1UL<<25);

  double *x, *y, *y_ref;
  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  cudaMallocHost((void**)&y_ref, N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) x[i] = drand48();

  Timer t;
  t.tic();
  scan(y_ref, x, N);
  double tt = t.toc();
  printf("CPU time = %f s\n", tt);

  double *x_d, *y_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));

  // dry run
  gpu_scan(y_d, x_d, N);

  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  t.tic();
  gpu_scan(y_d, x_d, N);
  cudaDeviceSynchronize();
  tt = t.toc();
  printf("GPU time = %f s\n", tt);
  printf("GPU bandwidth = %f GB/s\n", 4*N*sizeof(double)/tt/1e9);
  cudaMemcpyAsync(y, y_d, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  double err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, fabs(y[i]-y_ref[i]));
  printf("Error = %e\n", err/N);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(y_ref);

  return 0;
}

