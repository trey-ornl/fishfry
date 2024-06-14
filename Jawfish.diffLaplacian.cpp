#include "Jawfish.hpp"

#include <algorithm>

__global__ static void copyToSend(const int di, const int dj, const int dk, const double *const in, Jawfish::LapBuffers send)
{
  static_assert(Jawfish::lapNeighbors_ == 6,"assumes 6 Lapacian neighbors");

  const int p = threadIdx.x+blockIdx.x*blockDim.x;
  const int q = threadIdx.y+blockIdx.y*blockDim.y;
  const int dim = blockIdx.z;

  if ((dim == 0) && (p < dj) && (q < dk)) {

    const int j = p;
    const int k = q;
    send.buf[0][j+dj*k] = in[0+di*(j+dj*k)];
    send.buf[1][j+dj*k] = in[di-1+di*(j+dj*k)];

  } else if ((dim == 1) && (p < di) && (q < dk)) {

    const int i = p;
    const int k = q;
    send.buf[2][i+di*k] = in[i+di*(0+dj*k)];
    send.buf[3][i+di*k] = in[i+di*(dj-1+dj*k)];

  } else if ((dim == 2) && (p < di) && (q < dj)) {

    const int i = p;
    const int j = q;
    send.buf[4][i+di*j] = in[i+di*(j+dj*0)];
    send.buf[5][i+di*j] = in[i+di*(j+dj*(dk-1))];

  }
}

__device__ static inline void diffLapSide(const int i, const int j, const int k, const int di, const int dj, const int dk, const int dij, const double denomX, const double denomY, const double denomZ, const Jawfish::LapBuffers &recv, const double *const in, const double *const goal, double *const __restrict__ out)
{
  const int ijk = i+di*(j+dj*k);
  const double diag = -2.0*in[ijk];

  const double xm = (i > 0) ? in[ijk-1] : recv.buf[0][j+dj*k];
  const double xp = (i < di-1) ? in[ijk+1] : recv.buf[1][j+dj*k];
  const double lapX = denomX*(xm+xp+diag);

  const double ym = (j > 0) ? in[ijk-di] : recv.buf[2][i+di*k];
  const double yp = (j < dj-1) ? in[ijk+di] : recv.buf[3][i+di*k];
  const double lapY = denomY*(ym+yp+diag);
  
  const double zm = (k > 0) ? in[ijk-dij] : recv.buf[4][i+di*j];
  const double zp = (k < dk-1) ? in[ijk+dij] : recv.buf[5][i+di*j];
  const double lapZ = denomZ*(zm+zp+diag);

  out[ijk] = goal[ijk]-lapX-lapY-lapZ;
}

__global__ static void diffLapSides(const int di, const int dj, const int dk, const double denomX, const double denomY, const double denomZ, const Jawfish::LapBuffers recv, const double *const in, const double *const goal, double *const __restrict__ out)
{
  static_assert(Jawfish::lapNeighbors_ == 6,"assumes 6 Lapacian neighbors");

  const int dij = di*dj;
  const int p = threadIdx.x+blockIdx.x*blockDim.x;
  const int q = threadIdx.y+blockIdx.y*blockDim.y;
  const int dim = blockIdx.z;

  if ((dim == 0) && (p < dj) && (q < dk)) {

    ::diffLapSide(0,p,q,di,dj,dk,dij,denomX,denomY,denomZ,recv,in,goal,out);
    ::diffLapSide(di-1,p,q,di,dj,dk,dij,denomX,denomY,denomZ,recv,in,goal,out);

  } else if ((dim == 1) && (p > 0) && (p < di-1) && (q < dk)) {

    ::diffLapSide(p,0,q,di,dj,dk,dij,denomX,denomY,denomZ,recv,in,goal,out);
    ::diffLapSide(p,dj-1,q,di,dj,dk,dij,denomX,denomY,denomZ,recv,in,goal,out);

  } else if ((dim == 2) && (p > 0) && (p < di-1) && (q > 0) && (q < dj-1)) {

    ::diffLapSide(p,q,0,di,dj,dk,dij,denomX,denomY,denomZ,recv,in,goal,out);
    ::diffLapSide(p,q,dk-1,di,dj,dk,dij,denomX,denomY,denomZ,recv,in,goal,out);

  }
}

__global__ static void diffLapInner(const int di, const int dj, const int dk, const double denomX, const double denomY, const double denomZ, const double *const in, const double *const goal, double *const __restrict__ out)
{
  // +1 to avoid sides
  const int i = 1+threadIdx.x+blockIdx.x*blockDim.x;
  const int j = 1+threadIdx.y+blockIdx.y*blockDim.y;
  const int k = 1+threadIdx.z+blockIdx.z*blockDim.z;

  if ((i < di-1) && (j < dj-1) && (k < dk-1)) { // -1 to avoid sides
    const int ijk = i+di*(j+dj*k);
    const int dij = di*dj;
    const double diag = -2.0*in[ijk];
    const double lapX = denomX*(in[ijk-1]+in[ijk+1]+diag);
    const double lapY = denomY*(in[ijk-di]+in[ijk+di]+diag);
    const double lapZ = denomZ*(in[ijk-dij]+in[ijk+dij]+diag);
    out[ijk] = goal[ijk]-lapX-lapY-lapZ;
  }
}

void Jawfish::diffLaplacian(const double *const goal)
{
  for (int i = 0; i < lapNeighbors_; i++) MPI_Irecv(lapRecv_.buf[i],lapCount_[i],MPI_DOUBLE,lapRank_[i],lapRecvTag_[i],MPI_COMM_WORLD,&lapRecvReq_[i]);

  constexpr dim3 block2(16,16);
  const dim3 grid2((std::max(di_,dj_)-1)/block2.x+1,(std::max(dj_,dk_)-1)/block2.y+1,3);
  ::copyToSend<<<grid2,block2>>>(di_,dj_,dk_,dPhi_,lapSend_);
  CHECK(cudaEventRecord(event_,0));

  constexpr dim3 block3(8,8,8);
  const dim3 grid3((di_-3)/block3.x+1,(dj_-3)/block3.y+1,(dk_-3)/block3.z+1); // -2 to avoid sides
  ::diffLapInner<<<grid3,block3>>>(di_,dj_,dk_,denomX_,denomY_,denomZ_,dPhi_,goal,dRho_);

  CHECK(cudaEventSynchronize(event_));
  for (int i = 0; i < lapNeighbors_; i++) MPI_Isend(lapSend_.buf[i],lapCount_[i],MPI_DOUBLE,lapRank_[i],lapSendTag_[i],MPI_COMM_WORLD,&lapSendReq_[i]);

  MPI_Waitall(lapNeighbors_,lapRecvReq_,MPI_STATUSES_IGNORE);

  ::diffLapSides<<<grid2,block2>>>(di_,dj_,dk_,denomX_,denomY_,denomZ_,lapRecv_,dPhi_,goal,dRho_);

  MPI_Waitall(lapNeighbors_,lapSendReq_,MPI_STATUSES_IGNORE);
}
