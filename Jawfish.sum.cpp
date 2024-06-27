#include "Jawfish.hpp"

#include <algorithm>

__global__ static void sum(const long n, const double *const in, double *const __restrict__ allSum)
{
  constexpr int WARPSIZE = warpSize;
  constexpr int MAXWARPS = 1024/WARPSIZE;
  __shared__ double ourSum[MAXWARPS];

  const int thread = threadIdx.x;
  const int warp = thread/WARPSIZE;
  const bool warpRoot = (thread%WARPSIZE == 0);

  if (thread < MAXWARPS) ourSum[thread] = 0;
  __syncthreads();

  const long iLo = threadIdx.x+blockIdx.x*blockDim.x;
  const long stride = blockDim.x*gridDim.x;

  double mySum = 0;
  for (long i = iLo; i < n; i += stride) mySum += in[i];
  for (int i = 1; i < WARPSIZE; i += i) mySum += __shfl_down(mySum,i);
  if (warpRoot) ourSum[warp] = mySum;
  __syncthreads();

  if (thread < MAXWARPS) {
    mySum = ourSum[thread];
    for (int i = 1; i < MAXWARPS; i += i) mySum += __shfl_down(mySum,i);
    if (thread == 0) atomicAdd(allSum,mySum);
  }
}

double Jawfish::sum(const long n, const double *const in)
{
  CHECK(cudaMemset(sum_,0,sizeof(*sum_)));
  constexpr int block = 1024;
  const int grid = std::min(int((n-1)/block+1),threads_/block);
  ::sum<<<grid,block>>>(n,in,sum_);
  CHECK(cudaDeviceSynchronize());
  double allSum = 0;
  MPI_Allreduce(sum_,&allSum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  return allSum;
}

