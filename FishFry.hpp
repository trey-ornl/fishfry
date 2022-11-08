#pragma once

#include "gpu.hpp"
#include "TimeStamp.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mpi.h>

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
        printf("FishFry: %dx%dx%d points over %dx%dx%d MPI processes\n",nPoints[0],nPoints[1],nPoints[2],nTasks[0],nTasks[1],nTasks[2]);
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
      CHECK(hipMalloc(&f_,bytes_));
      CHECK(hipMalloc(&phi_,bytes_));
      CHECK(hipMalloc(&diffs_,3*sizeof(double)));

      CHECK(hipMemsetAsync(diffs_,0,3*sizeof(double),0));
      CHECK(hipMemsetAsync(phi_,0,bytes_));
      diff<<<nb_,nt_>>>(ni_,nj_,nk_,xLo_,yLo_,zLo_,dx_,dy_,dz_,phi_,diffs_);
      CHECK(hipGetLastError());
      CHECK(hipDeviceSynchronize());

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
      CHECK(hipFree(diffs_));
      CHECK(hipFree(phi_));
      CHECK(hipFree(f_));
    }

    void run(const int nIters)
    {
      std::vector<TimeStamp> totalStamps;


      for (int i = 0; i <= nIters; i++) {
        init<<<nb_,nt_>>>(ni_,nj_,nk_,xLo_,yLo_,zLo_,dx_,dy_,dz_,f_);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
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

        CHECK(hipMemsetAsync(diffs_,0,3*sizeof(double),0));
        diff<<<nb_,nt_>>>(ni_,nj_,nk_,xLo_,yLo_,zLo_,dx_,dy_,dz_,phi_,diffs_);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());

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

    __global__ static void init(const int ni, const int nj, const int nk, const double xLo, const double yLo, const double zLo, const double dx, const double dy, const double dz, double *const f)
    {
      const int i = threadIdx.z+blockDim.z*blockIdx.z;
      const int j = threadIdx.y+blockDim.y*blockIdx.y;
      const int k = threadIdx.x+blockDim.x*blockIdx.x;

      if ((i >= ni) || (j >= nj) || (k >= nk)) return;

      const double x = xLo+i*dx;
      const double cx = cos(x);
      const double sx = sin(x);
      const double ex = exp(sx);
      const double px = cx*ex;
      const double dpx = -ex*sx*cx*(sx+3.0);

      const double y = yLo+j*dy;
      const double cy = cos(y);
      const double sy = sin(y);
      const double ey = exp(cy);
      const double py = sy*ey;
      const double dpy = sy*ey*(sy*sy-3.0*cy-1.0);

      const double z = zLo+k*dz;
      const double cz = cos(z);
      const double sz = sin(z);
      const double ez = exp(cz);
      const double pz = cz*ez;
      const double dpz = ez*(3.0*sz*sz-cz*cz*cz-1.0);

      const int ijk = (i*nj+j)*nk+k;
      f[ijk] = px*py*dpz+px*dpy*pz+dpx*py*pz;
    }

    __device__ static double atomicMax(double *const address, const double val)
    {
      double result = 0;
      unsigned long long *p = reinterpret_cast<unsigned long long *>(&result);
      *p = ::atomicMax(reinterpret_cast<unsigned long long *>(address),*reinterpret_cast<const unsigned long long *>(&val));
      return result;
    }

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
        const double cx = cos(x);
        const double sx = sin(x);
        const double ex = exp(sx);
        const double px = cx*ex;

        const double y = yLo+j*dy;
        const double cy = cos(y);
        const double sy = sin(y);
        const double ey = exp(cy);
        const double py = sy*ey;

        const double z = zLo+k*dz;
        const double cz = cos(z);
        const double ez = exp(cz);
        const double pz = cz*ez;

        const int ijk = (i*nj+j)*nk+k;
        const double dPhi = fabs(px*py*pz-phi[ijk]);

        atomicAdd(shared,dPhi);
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

};
