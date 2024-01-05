#pragma once

#include <cassert>
#include <climits>
#include <cstdio>

#ifdef O_HIP

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#ifdef __HIPCC__
#define __CUDACC__
#endif

#define CUFFT_D2Z HIPFFT_D2Z
#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INVERSE HIPFFT_BACKWARD
#define CUFFT_Z2D HIPFFT_Z2D
#define CUFFT_Z2Z HIPFFT_Z2Z

#define cufftDestroy hipfftDestroy
#define cufftDoubleComplex hipfftDoubleComplex
#define cufftExecD2Z hipfftExecD2Z
#define cufftExecZ2D hipfftExecZ2D
#define cufftExecZ2Z hipfftExecZ2Z
#define cufftHandle hipfftHandle
#define cufftPlanMany hipfftPlanMany

#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetLastError hipGetLastError
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaSetDevice hipSetDevice

static void __attribute__((unused)) check(const hipfftResult err, const char *const file, const int line)
{
  if (err == HIPFFT_SUCCESS) return;
  fprintf(stderr,"HIPFFT ERROR AT LINE %d OF FILE '%s': %d\n",line,file,err);
  fflush(stderr);
  exit(err);
}

static void __attribute__((unused)) check(const hipError_t err, const char *const file, const int line)
{
  if (err == hipSuccess) return;
  fprintf(stderr,"HIP ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,hipGetErrorName(err),hipGetErrorString(err));
  fflush(stderr);
  exit(err);
}

#else

#include <cufft.h>
#include <cuda_runtime.h>

#define hipLaunchKernelGGL(F,G,B,M,S,...) F<<<G,B,M,S>>>(__VA_ARGS__)

static void check(const cufftResult err, const char *const file, const int line)
{
  if (err == CUFFT_SUCCESS) return;
  fprintf(stderr,"CUFFT ERROR AT LINE %d OF FILE '%s': %d\n",line,file,err);
  fflush(stderr);
  exit(err);
}

static void check(const cudaError_t err, const char *const file, const int line)
{
  if (err == cudaSuccess) return;
  fprintf(stderr,"CUDA ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,cudaGetErrorName(err),cudaGetErrorString(err));
  fflush(stderr);
  exit(err);
}

#endif

#define CHECK(X) check(X,__FILE__,__LINE__)

#if defined(__CUDACC__) || defined(__HIPCC__)

static constexpr int GPU_MAX_THREADS = 64;

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun0x2(const F f)
{
  const int i0 = threadIdx.y;
  const int i1 = threadIdx.x;
  f(i0,i1);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun1x1(const F f)
{
  const int i0 = blockIdx.x;
  const int i1 = threadIdx.x;
  f(i0,i1);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun2x0(const int n1, const F f)
{
  const int i0 = blockIdx.y;
  const int i1 = blockIdx.x*blockDim.x+threadIdx.x;
  if (i1 < n1) f(i0,i1);
}

template <typename F>
void gpuFor(const int n0, const int n1, const F f)
{
  //printf("gpuFor(%d,%d)->",n0,n1);
  if ((n0 <= 0) || (n1 <= 0)) return;
  const long nl01 = long(n0)*long(n1);
  assert(nl01 < long(INT_MAX));

  if (n1 > GPU_MAX_THREADS) {
    const int b1 = (n1+GPU_MAX_THREADS-1)/GPU_MAX_THREADS;
    const int t1 = (n1+b1-1)/b1;
    //printf("2x0 (%d,%d)x%d(%d)\n",b1,n0,t1,n1);
    gpuRun2x0<<<dim3(b1,n0),dim3(t1)>>>(n1,f);
    CHECK(cudaGetLastError());
  } else if (nl01 > GPU_MAX_THREADS) {
    //printf("1x1 %dx%d\n",n0,n1);
    gpuRun1x1<<<n0,n1>>>(f);
    CHECK(cudaGetLastError());
  } else {
    //printf("0x2 1x(%d,%d)\n",n1,n0);
    gpuRun0x2<<<1,dim3(n1,n0)>>>(f);
    CHECK(cudaGetLastError());
  }
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun0x3(const F f)
{
  const int i0 = threadIdx.z;
  const int i1 = threadIdx.y;
  const int i2 = threadIdx.x;
  f(i0,i1,i2);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun1x2(const F f)
{
  const int i0 = blockIdx.x;
  const int i1 = threadIdx.y;
  const int i2 = threadIdx.x;
  f(i0,i1,i2);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun2x1(const F f)
{
  const int i0 = blockIdx.y;
  const int i1 = blockIdx.x;
  const int i2 = threadIdx.x;
  f(i0,i1,i2);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun3x0(const int n2, const F f)
{
  const int i0 = blockIdx.z;
  const int i1 = blockIdx.y;
  const int i2 = blockIdx.x*blockDim.x+threadIdx.x;
  if (i2 < n2) f(i0,i1,i2);
}

template <typename F>
void gpuFor(const int n0, const int n1, const int n2, const F f)
{
  //printf("gpuFor(%d,%d,%d)->",n0,n1,n2);
  if ((n0 <= 0) || (n1 <= 0) || (n2 <= 0)) return;
  const long nl12 = long(n1)*long(n2);
  const long nl012 = long(n0)*nl12;
  assert(nl012 < long(INT_MAX));

  if (n2 > GPU_MAX_THREADS) {
    const int b2 = (n2+GPU_MAX_THREADS-1)/GPU_MAX_THREADS;
    const int t2 = (n2+b2-1)/b2;
    //printf("3x0 (%d,%d,%d)x%d(%d)\n",b2,n1,n0,t2,n2);
    gpuRun3x0<<<dim3(b2,n1,n0),t2>>>(n2,f);
    CHECK(cudaGetLastError());
  } else if (nl12 > GPU_MAX_THREADS) {
    //printf("2x1 (%d,%d)x%d\n",n1,n0,n2);
    gpuRun2x1<<<dim3(n1,n0),n2>>>(f);
    CHECK(cudaGetLastError());
  } else if (nl012 > GPU_MAX_THREADS) {
    //printf("1x2 %dx(%d,%d)\n",n0,n2,n1);
    gpuRun1x2<<<n0,dim3(n2,n1)>>>(f);
    CHECK(cudaGetLastError());
  } else {
    //printf("0x3 1x(%d,%d,%d)\n",n2,n1,n0);
    gpuRun0x3<<<1,dim3(n2,n1,n0)>>>(f);
    CHECK(cudaGetLastError());
  }
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun1x3(const F f)
{
  const int i0 = blockIdx.x;
  const int i1 = threadIdx.z;
  const int i2 = threadIdx.y;
  const int i3 = threadIdx.x;
  f(i0,i1,i2,i3);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun2x2(const F f)
{
  const int i0 = blockIdx.y;
  const int i1 = blockIdx.x;
  const int i2 = threadIdx.y;
  const int i3 = threadIdx.x;
  f(i0,i1,i2,i3);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun3x1(const F f)
{
  const int i0 = blockIdx.z;
  const int i1 = blockIdx.y;
  const int i2 = blockIdx.x;
  const int i3 = threadIdx.x;
  f(i0,i1,i2,i3);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun4x0(const int n23, const int n3, const F f)
{
  const int i23 = blockIdx.x*blockDim.x+threadIdx.x;
  if (i23 < n23) {
    const int i0 = blockIdx.z;
    const int i1 = blockIdx.y;
    const int i2 = i23/n3;
    const int i3 = i23%n3;
    f(i0,i1,i2,i3);
  }
}

template <typename F>
void gpuFor(const int n0, const int n1, const int n2, const int n3, const F f)
{
  //printf("gpuFor(%d,%d,%d,%d)->",n0,n1,n2,n3);
  if ((n0 <= 0) || (n1 <= 0) || (n2 <= 0) || (n3 <= 0)) return;
  const long nl23 = long(n2)*long(n3);
  const long nl123 = long(n1)*nl23;
  assert(long(n0)*nl123 < long(INT_MAX));

  const int n23 = int(nl23);
  const int n123 = int(nl123);
  if (n3 > GPU_MAX_THREADS) {
    const int b23 = (n23+GPU_MAX_THREADS-1)/GPU_MAX_THREADS;
    const int t23 = (n23+b23-1)/b23;
    //printf("4x0 (%d,%d,%d)x%d(%d,%d)\n",b23,n1,n0,t23,n23,n3); 
    gpuRun4x0<<<dim3(b23,n1,n0),t23>>>(n23,n3,f);
    CHECK(cudaGetLastError());
  } else if (n23 > GPU_MAX_THREADS) {
    //printf("3x1 (%d,%d,%d)x(%d)\n",n2,n1,n0,n3);
    gpuRun3x1<<<dim3(n2,n1,n0),n3>>>(f);
    CHECK(cudaGetLastError());
  } else if (n123 > GPU_MAX_THREADS) {
    //printf("2x2 (%d,%d)x(%d,%d)\n",n1,n0,n3,n2);
    gpuRun2x2<<<dim3(n1,n0),dim3(n3,n2)>>>(f);
    CHECK(cudaGetLastError());
  } else {
    //printf("1x3 (%d)x(%d,%d,%d)\n",n0,n3,n2,n1);
    gpuRun1x3<<<n0,dim3(n3,n2,n1)>>>(f);
    CHECK(cudaGetLastError());
  }
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun2x3(const F f)
{
  const int i0 = blockIdx.y;
  const int i1 = blockIdx.x;
  const int i2 = threadIdx.z;
  const int i3 = threadIdx.y;
  const int i4 = threadIdx.x;
  f(i0,i1,i2,i3,i4);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun3x2(const F f)
{
  const int i0 = blockIdx.z;
  const int i1 = blockIdx.y;
  const int i2 = blockIdx.x;
  const int i3 = threadIdx.y;
  const int i4 = threadIdx.x;
  f(i0,i1,i2,i3,i4);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun4x1(const int n1, const F f)
{
  const int i01 = blockIdx.z;
  const int i0 = i01/n1;
  const int i1 = i01%n1;
  const int i2 = blockIdx.y;
  const int i3 = blockIdx.x;
  const int i4 = threadIdx.x;
  f(i0,i1,i2,i3,i4);
}

template <typename F>
__global__ __launch_bounds__(GPU_MAX_THREADS)
void gpuRun5x0(const int n1, const int n34, const int n4, const F f)
{
  const int i34 = blockIdx.x*blockDim.x+threadIdx.x;
  if (i34 < n34) {
    const int i01 = blockIdx.z;
    const int i0 = i01/n1;
    const int i1 = i01%n1;
    const int i2 = blockIdx.y;
    const int i3 = i34/n4;
    const int i4 = i34%n4;
    f(i0,i1,i2,i3,i4);
  }
}

template <typename F>
void gpuFor(const int n0, const int n1, const int n2, const int n3, const int n4, const F f)
{
  //printf("gpuFor(%d,%d,%d,%d,%d)->",n0,n1,n2,n3,n4);
  if ((n0 <= 0) || (n1 <= 0) || (n2 <= 0) || (n3 <= 0) || (n4 <= 0)) return;
  const long nl01 = long(n0)*long(n1);
  const long nl34 = long(n3)*long(n4);
  assert(nl01*long(n2)*nl34 < long(INT_MAX));

  const int n34 = int(nl34);
  if (n4 > GPU_MAX_THREADS) {
    const int n01 = int(nl01);
    const int b34 = (n34+GPU_MAX_THREADS-1)/GPU_MAX_THREADS;
    const int t34 = (n34+b34-1)/b34;
    //printf("5x0 (%d,%d,%d)x%d(%d,%d,%d)\n",b34,n2,n01,t34,n1,n34,n4); 
    gpuRun5x0<<<dim3(b34,n2,n01),t34>>>(n1,n34,n4,f);
    CHECK(cudaGetLastError());
  } else if (n34 > GPU_MAX_THREADS) {
    const int n01 = n0*n1;
    //printf("4x1 (%d,%d,%d)x%d(%d)\n",n3,n2,n01,n4,n1);
    gpuRun4x1<<<dim3(n3,n2,n01),n4>>>(n1,f);
    CHECK(cudaGetLastError());
  } else if (n2*n34 > GPU_MAX_THREADS) {
    //printf("3x2 (%d,%d,%d)x(%d,%d)\n",n2,n1,n0,n4,n3);
    gpuRun3x2<<<dim3(n2,n1,n0),dim3(n4,n3)>>>(f);
    CHECK(cudaGetLastError());
  } else {
    //printf("2x3 (%d,%d)x(%d,%d,%d)\n",n1,n0,n4,n3,n2);
    gpuRun2x3<<<dim3(n1,n0),dim3(n4,n3,n2)>>>(f);
    CHECK(cudaGetLastError());
  }
}

#define GPU_LAMBDA [=] __device__

#endif
