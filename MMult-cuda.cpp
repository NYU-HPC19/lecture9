#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "utils.h"

void MMult0(long m, long n, long k, double *a, double *b, double *c) {
#pragma omp parallel for
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1(long m, long n, long k, double *A, double *B, double *C) {
  double *A_d, *B_d, *C_d;
  cudaMalloc((void**)&A_d, m*k*sizeof(double));
  cudaMalloc((void**)&B_d, k*n*sizeof(double));
  cudaMalloc((void**)&C_d, m*n*sizeof(double));

  cudaMemcpy(A_d, A, m*k*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, k*n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C, m*n*sizeof(double), cudaMemcpyHostToDevice);

  const double alpha = 1.0;
  const double beta = 1.0;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_d, m, B_d, k, &beta, C_d, m);
  cublasDestroy(handle);

  cudaMemcpy(C, C_d, m*n*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main(int argc, char** argv) {
  const long PFIRST = 256;
  const long PLAST = 4096;
  const long PINC = 256;

  Timer t;
  printf(" Dimension       Time    Gflop/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = 2.0*m*n*k*NREPEATS*1e-9/time;
    printf("%10d %10f %10f", p, time, flops);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    free(a);
    free(b);
    free(c);
  }

  return 0;
}

