#pragma once

#include "gpu.hpp"
#include "TimeStamp.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mpi.h>
#include <typeinfo>

__device__ static inline double ddf(const double a, const double x)
{
  const double c = cos(x);
  const double s = sin(x);
  const double a2 = a*a;
  const double a4 = a2*a2;
  return a*exp(a2*(c-1.0))*s*(a4*s*s-3.0*a2*c-1.0);
}

__device__ static inline double initRho(const double x, const double y, const double z)
{
  return ddf(1.0,x+0.9)+ddf(1.0,y+0.99)+ddf(1.0,z+1.1)-ddf(10.0,x-1.2)-ddf(10.0,y-1.1)-ddf(10.0,z-1.01);
}

__device__ static inline double f(const double a, const double x)
{
  return a*sin(x)*exp(a*a*(cos(x)-1.0));
}

__device__ static inline double initPhi(const double x, const double y, const double z)
{
  return f(1.0,x+0.9)+f(1.0,y+0.99)+f(1.0,z+1.1)-f(10.0,x-1.2)-f(10.0,y-1.1)-f(10.0,z-1.01);
}

__global__ static void initRho(const int ni, const int nj, const int nk, const double xLo, const double yLo, const double zLo, const double dx, const double dy, const double dz, double *const rho)
{
  const int i = threadIdx.z+blockDim.z*blockIdx.z;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.x+blockDim.x*blockIdx.x;

  if ((i >= ni) || (j >= nj) || (k >= nk)) return;

  const double x = xLo+i*dx;
  const double y = yLo+j*dy;
  const double z = zLo+k*dz;

  const int ijk = (i*nj+j)*nk+k;
#ifdef STENCIL_7PT
  const double phi = initPhi(x,y,z);
  const double ddx = (initPhi(x-dx,y,z)+initPhi(x+dx,y,z)-2.0*phi)/(dx*dx);
  const double ddy = (initPhi(x,y-dy,z)+initPhi(x,y+dy,z)-2.0*phi)/(dy*dy);
  const double ddz = (initPhi(x,y,z-dz)+initPhi(x,y,z+dz)-2.0*phi)/(dz*dz);
  rho[ijk] = ddx+ddy+ddz;
#else
  rho[ijk] = initRho(x,y,z);
#endif
}

#ifndef __HIPCC__
__device__ static double atomicMax(double *const address, const double val)
{
  double result = 0;
  unsigned long long *p = reinterpret_cast<unsigned long long *>(&result);
  *p = ::atomicMax(reinterpret_cast<unsigned long long *>(address),*reinterpret_cast<const unsigned long long *>(&val));
  return result;
}
#endif

__global__ static void diff(const int ni, const int nj, const int nk, const double xLo, const double yLo, const double zLo, const double dx, const double dy, const double dz, const double *const phi, double *const diffs)
{
  __shared__ double shared[3];

  const bool root = ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0));
  if (root) shared[0] = shared[1] = shared[2] = 0.0;
  __syncthreads();

  const int i = threadIdx.z+blockDim.z*blockIdx.z;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.x+blockDim.x*blockIdx.x;

  if ((i < ni) && (j < nj) && (k < nk)) {

    const double x = xLo+i*dx;
    const double y = yLo+j*dy;
    const double z = zLo+k*dz;
    const int ijk = (i*nj+j)*nk+k;
    const double dPhi = fabs(initPhi(x,y,z)-phi[ijk]);

    atomicAdd(&shared[0],dPhi);
    atomicAdd(shared+1,dPhi*dPhi);
    atomicMax(shared+2,dPhi);
  }
  __syncthreads();
  if (root) {
    atomicAdd(diffs,shared[0]);
    atomicAdd(diffs+1,shared[1]);
    atomicMax(diffs+2,shared[2]);
  }
}

template <typename P>
class FishFry
{
  public:

    FishFry(const int nTasks[3], const int nPoints[3]):
      dx_(2.0*M_PI/nPoints[0]),
      dy_(2.0*M_PI/nPoints[1]),
      dz_(2.0*M_PI/nPoints[2]),
      ni_((nPoints[0]+nTasks[0]-1)/nTasks[0]),
      nj_((nPoints[1]+nTasks[1]-1)/nTasks[1]),
      nk_((nPoints[2]+nTasks[2]-1)/nTasks[2]),
      nb_((nk_+nt_.x-1)/nt_.x,(nj_+nt_.y-1)/nt_.y,(ni_+nt_.z-1)/nt_.z),
      totalPoints_(double(nPoints[0])*double(nPoints[1])*double(nPoints[2])),
      rank_(MPI_PROC_NULL),
      xLo_(0.0), yLo_(0.0), zLo_(0.0),
      bytes_(0),
      f_(nullptr), phi_(nullptr), diffs_(nullptr),
      scales_{0.0,0.0,0.0}
    {
      int size = 0;
      MPI_Comm_size(MPI_COMM_WORLD,&size);
      assert(size == nTasks[0]*nTasks[1]*nTasks[2]);

      MPI_Comm_rank(MPI_COMM_WORLD,&rank_);
      if (rank_ == 0) {
        printf("FishFry<%s>: %dx%dx%d points over %dx%dx%d MPI processes",typeid(*poisson_).name(),nPoints[0],nPoints[1],nPoints[2],nTasks[0],nTasks[1],nTasks[2]);
#ifdef STENCIL_7PT
        printf(", 7-point Laplacian\n");
#else
        printf(", spectral Laplacian\n");
#endif
        fflush(stdout);
      }
      const int id[] = { rank_/(nTasks[1]*nTasks[2]), (rank_/nTasks[2])%nTasks[1], rank_%nTasks[2] };
      const double lo[] = { -M_PI, -M_PI, -M_PI };
      const double hi[] = { M_PI-dx_, M_PI-dy_, M_PI-dz_ };
      poisson_.reset(new P(nPoints,lo,hi,nTasks,id));

      xLo_ = lo[0]+id[0]*ni_*dx_;
      yLo_ = lo[1]+id[1]*nj_*dy_;
      zLo_ = lo[2]+id[2]*nk_*dz_;

      bytes_ = std::max(poisson_->bytes(),ni_*nj_*nk_*sizeof(double));
      CHECK(cudaMalloc(&f_,bytes_));
      CHECK(cudaMalloc(&phi_,bytes_));
      CHECK(cudaMalloc(&diffs_,3*sizeof(double)));

      CHECK(cudaMemsetAsync(diffs_,0,3*sizeof(double),0));
      CHECK(cudaMemsetAsync(phi_,0,bytes_));
      diff<<<nb_,nt_>>>(ni_,nj_,nk_,xLo_,yLo_,zLo_,dx_,dy_,dz_,phi_,diffs_);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());

      double allDiffs[3];
      MPI_Reduce(diffs_,allDiffs,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Reduce(diffs_+2,allDiffs+2,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      if (rank_ == 0) {
        scales_[0] = 1.0/allDiffs[0];
        scales_[1] = 1.0/allDiffs[1];
        scales_[2] = 1.0/allDiffs[2];
      }
    }

    ~FishFry()
    {
      CHECK(cudaFree(diffs_));
      CHECK(cudaFree(phi_));
      CHECK(cudaFree(f_));
    }

    void run(const int nIters)
    {
      std::vector<TimeStamp> totalStamps;

      for (int i = 0; i <= nIters; i++) {
        ::initRho<<<nb_,nt_>>>(ni_,nj_,nk_,xLo_,yLo_,zLo_,dx_,dy_,dz_,f_);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<TimeStamp> stamps;
        poisson_->solve(bytes_,f_,phi_,stamps);
        const double time = stamps.back().second-stamps.front().second;
        if (i == 0) {
          totalStamps = stamps;
          for (unsigned j = 0; j < stamps.size(); j++) totalStamps.at(j).second = 0;
        } else {
          for (unsigned j = 0; j < stamps.size(); j++) totalStamps.at(j).second += stamps.at(j).second;
        }

        CHECK(cudaMemsetAsync(diffs_,0,3*sizeof(double),0));
        ::diff<<<nb_,nt_>>>(ni_,nj_,nk_,xLo_,yLo_,zLo_,dx_,dy_,dz_,phi_,diffs_);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        double allDiffs[3];
        MPI_Reduce(diffs_,allDiffs,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(diffs_+2,allDiffs+2,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if (rank_ == 0) {
          printf("%d: Time %10.4e Rate %10.4e L_1 %10.4e L_2 %10.4e L_\\inf %10.4e",i,time,totalPoints_/time,allDiffs[0]*scales_[0],sqrt(allDiffs[1]*scales_[1]),allDiffs[2]*scales_[2]);
          if (i == 0) printf(" (warmup, ignored)");
          printf("\n");
          fflush(stdout);
        }
      }

      if (rank_ == 0) {
        printf("\nTimes Averaged across Iterations\n");
        printf("Min Rank   | Avg Rank   | Max Rank   (%%Total) | Phase\n");
        printf("--------------------------------------------------------\n");
      }

      int size = 0;
      MPI_Comm_size(MPI_COMM_WORLD,&size);
      const double perSize = 1.0/double(size);

      const double perIter = 1.0/double(nIters);
      double percent, t, tMax, tMin, tSum;
      for (unsigned i = 0; i < totalStamps.size(); i++) {
        const double dt = (i == 0) ? totalStamps.back().second-totalStamps.front().second : totalStamps[i].second-totalStamps[i-1].second;
        t = perIter*dt;
        MPI_Reduce(&t,&tMax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t,&tMin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
        MPI_Reduce(&t,&tSum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if (i == 0) percent = 100.0/tMax;
        if (rank_ == 0) {
          printf("%10.4e | %10.4e | %10.4e (%5.1f%%) | %s\n",tMin,tSum*perSize,tMax,tMax*percent,totalStamps[i].first.c_str());
        }
      }
      if (rank_ == 0) fflush(stdout);
    }

  protected:
    static constexpr dim3 nt_{8,8,8};
    double dx_,dy_,dz_;
    int ni_,nj_,nk_;
    dim3 nb_;
    double totalPoints_;
    int rank_;
    std::unique_ptr<P> poisson_;
    double xLo_,yLo_,zLo_;
    size_t bytes_;
    double *f_;
    double *phi_;
    double *diffs_;
    double scales_[3];

};
