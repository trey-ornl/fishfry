#include "Jawfish.hpp"

#include <algorithm>
#include <cassert>

struct SmoothRecvBuffers { double *__restrict__ buf[3][2][3][3]; };
struct SmoothSendBuffers { double *__restrict__ buf[3]; };
struct SmoothSides { double *__restrict__ side[3][2]; };

static __global__ void copyFromBuf(
  const int diMax, const int djMax, const int dkMax,
  SmoothRecvBuffers recv, SmoothSides out)
{
  const int p = threadIdx.x+blockIdx.x*blockDim.x;
  const int q = threadIdx.y+blockIdx.y*blockDim.y;
  const int dim = blockIdx.z;
  const int dir = threadIdx.z;

  const int dp = (dim == 0) ? djMax : diMax;
  const int dq = (dim == 2) ? djMax : dkMax;

  if ((p < dp) && (q < dq)) {
    for (int bq = 0; bq < 3; bq++) {
      for (int bp = 0; bp < 3; bp++) {
        out.side[dim][dir][p+bp*dp+3*dp*(q+bq*dq)] = recv.buf[dim][dir][bq][bp][p+dp*q];
      }
    }
  }
}

static __global__ void copyToBuf(
  const int di, const int dj, const int dk,
  const int diMax, const int djMax,
  const double *const in, const SmoothSendBuffers send)
{
  const int p = threadIdx.x+blockIdx.x*blockDim.x;
  const int q = threadIdx.y+blockIdx.y*blockDim.y;
  const int dim = blockIdx.z;

  if ((dim == 0) && (p < dj) && (q < dk)) {
    send.buf[0][p+djMax*q] = in[di-1+di*(p+dj*q)];
  }
  else if ((dim == 1) && (p < di) && (q < dk)) {
    send.buf[1][p+diMax*q] = in[p+di*(dj-1+dj*q)];
  }
  else if ((dim == 2) && (p < di) && (q < dj)) {
    send.buf[2][p+diMax*q] = in[p+di*(q+dj*(dk-1))];
  }
}

static __global__ void smooth(
  const int di, const int dj, const int dk,
  const int diMax, const int djMax, const int dkMax,
  const int radius, const double *const smoother,
  const SmoothSides in, double *const __restrict__ inout)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= di) || (j >= dj) || (k >= dk)) return;

  const int diam = radius+radius+1;
  double sum = 0;

  if (k < radius)
    for (int jr = -radius; jr <= radius; jr++)  
      for (int ir = -radius; ir <= radius; ir++) {
        sum += in.side[2][0][i+ir+diMax+3*diMax*(j+jr+djMax)]*
          smoother[ir+radius+diam*(jr+radius+diam*(k+1+radius))];
      }

  if (k+radius >= dk-1)
    for (int jr = -radius; jr <= radius; jr++)  
      for (int ir = -radius; ir <= radius; ir++) {
        sum += in.side[2][1][i+ir+diMax+3*diMax*(j+jr+djMax)]*
          smoother[ir+radius+diam*(jr+radius+diam*(dk-1-k+radius))];
      }

  if (j < radius)
    for (int kr = -radius; kr <= radius; kr++)  
      if ((k+kr != -1) && (k+kr != dk-1))
        for (int ir = -radius; ir <= radius; ir++) {
          sum += in.side[1][0][i+ir+diMax+3*diMax*(k+kr+dkMax)]*
            smoother[ir+radius+diam*(j+1+radius+diam*(kr+radius))];
        }

  if (j+radius >= dj-1)
    for (int kr = -radius; kr <= radius; kr++)  
      if ((k+kr != -1) && (k+kr != dk-1))
        for (int ir = -radius; ir <= radius; ir++) {
          sum += in.side[1][1][i+ir+diMax+3*diMax*(k+kr+dkMax)]*
            smoother[ir+radius+diam*(dj-1-j+radius+diam*(kr+radius))];
        }

  if (i < radius)
    for (int kr = -radius; kr <= radius; kr++)  
      if ((k+kr != -1) && (k+kr != dk-1))
        for (int jr = -radius; jr <= radius; jr++)
          if ((j+jr != -1) && (j+jr != dj-1)) {
            sum += in.side[0][0][j+jr+djMax+3*djMax*(k+kr+dkMax)]*
              smoother[i+1+radius+diam*(jr+radius+diam*(kr+radius))];
          }

  if (i+radius >= di-1)
    for (int kr = -radius; kr <= radius; kr++)  
      if ((k+kr != -1) && (k+kr != dk-1))
        for (int jr = -radius; jr <= radius; jr++)
          if ((j+jr != -1) && (j+jr != dj-1)) {
            sum += in.side[0][1][j+jr+djMax+3*djMax*(k+kr+dkMax)]*
              smoother[di-1-i+radius+diam*(jr+radius+diam*(kr+radius))];
          }

  if (sum) inout[i+di*(j+dj*k)] += sum;
}

void Jawfish::smoothSides()
{
  constexpr int smoothMsgs = 3*3*2*3;
  SmoothRecvBuffers smoothRecv;
  double *smoothRecvBuf[smoothMsgs];
  SmoothSendBuffers smoothSend;
  double *smoothSendBuf[smoothMsgs];
  SmoothSides smoothIn;
  int smoothCount[smoothMsgs], smoothSendRank[smoothMsgs], smoothRecvRank[smoothMsgs];
  int smoothTag[smoothMsgs];
  MPI_Request smoothRecvReq[smoothMsgs], smoothSendReq[smoothMsgs];

  int tag = 133;
  int id = 0;

  { // X
    const int count = djMax_*dkMax_;
    const long bytes = count*sizeof(double);
    CHECK(cudaMalloc((void**)&smoothSend.buf[0],bytes));
    CHECK(cudaMemset(smoothSend.buf[0],0,bytes));
    for (int i = -1; i < 1; i++) {
      for (int k = -1; k < 2; k++) {
        for (int j = -1; j < 2; j++) {
          smoothSendBuf[id] = smoothSend.buf[0];
          CHECK(cudaMalloc(&smoothRecvBuf[id],bytes));
          smoothRecv.buf[0][i+1][k+1][j+1] = smoothRecvBuf[id];
          CHECK(cudaMemset(smoothRecvBuf[id],0,bytes));
          smoothRecvRank[id] = neighbor(i,j,k);
          smoothSendRank[id] = neighbor(-i,-j,-k);
          smoothCount[id] = count;
          smoothTag[id] = tag++;
          id++;
        }
      }
    }
    const long sideBytes = 3*3*bytes;
    CHECK(cudaMalloc((void**)&smoothIn.side[0][0],sideBytes));
    CHECK(cudaMalloc((void**)&smoothIn.side[0][1],sideBytes));
    CHECK(cudaMemset(smoothIn.side[0][0],0,sideBytes));
    CHECK(cudaMemset(smoothIn.side[0][1],0,sideBytes));
  }
  { // Y
    const int count = diMax_*dkMax_;
    const long bytes = count*sizeof(double);
    CHECK(cudaMalloc((void**)&smoothSend.buf[1],bytes));
    CHECK(cudaMemset(smoothSend.buf[1],0,bytes));
    for (int j = -1; j < 1; j++) {
      for (int k = -1; k < 2; k++) {
        for (int i = -1; i < 2; i++) {
          smoothSendBuf[id] = smoothSend.buf[1];
          CHECK(cudaMalloc(&smoothRecvBuf[id],bytes));
          smoothRecv.buf[1][j+1][k+1][i+1] = smoothRecvBuf[id];
          CHECK(cudaMemset(smoothRecvBuf[id],0,bytes));
          smoothRecvRank[id] = neighbor(i,j,k);
          smoothSendRank[id] = neighbor(-i,-j,-k);
          smoothCount[id] = count;
          smoothTag[id] = tag++;
          id++;
        }
      }
    }
    const long sideBytes = 3*3*bytes;
    CHECK(cudaMalloc((void**)&smoothIn.side[1][0],sideBytes));
    CHECK(cudaMalloc((void**)&smoothIn.side[1][1],sideBytes));
    CHECK(cudaMemset(smoothIn.side[1][0],0,sideBytes));
    CHECK(cudaMemset(smoothIn.side[1][1],0,sideBytes));
  }
  { // Z
    const int count = diMax_*djMax_;
    const long bytes = count*sizeof(double);
    CHECK(cudaMalloc((void**)&smoothSend.buf[2],bytes));
    CHECK(cudaMemset(smoothSend.buf[2],0,bytes));
    for (int k = -1; k < 1; k++) {
      for (int j = -1; j < 2; j++) {
        for (int i = -1; i < 2; i++) {
          smoothSendBuf[id] = smoothSend.buf[2];
          CHECK(cudaMalloc(&smoothRecvBuf[id],bytes));
          smoothRecv.buf[2][k+1][j+1][i+1] = smoothRecvBuf[id];
          CHECK(cudaMemset(smoothRecvBuf[id],0,bytes));
          smoothRecvRank[id] = neighbor(i,j,k);
          smoothSendRank[id] = neighbor(-i,-j,-k);
          smoothCount[id] = count;
          smoothTag[id] = tag++;
          id++;
        }
      }
    }
    const long sideBytes = 3*3*bytes;
    CHECK(cudaMalloc((void**)&smoothIn.side[2][0],sideBytes));
    CHECK(cudaMalloc((void**)&smoothIn.side[2][1],sideBytes));
    CHECK(cudaMemset(smoothIn.side[2][0],0,sideBytes));
    CHECK(cudaMemset(smoothIn.side[2][1],0,sideBytes));
    CHECK(cudaDeviceSynchronize());
  }
  assert(id == smoothMsgs);

  for (int id = 0; id < smoothMsgs; id++)
    MPI_Irecv(smoothRecvBuf[id],smoothCount[id],MPI_DOUBLE,smoothRecvRank[id],smoothTag[id],MPI_COMM_WORLD,&smoothRecvReq[id]);

  {
    constexpr dim3 block(32,8);
    const dim3 grid((std::max(di_,dj_)-1)/block.x+1,(std::max(dj_,dk_)-1)/block.y+1,3);
    ::copyToBuf<<<grid,block>>>(di_,dj_,dk_,diMax_,djMax_,dRho_,smoothSend);
  }

  CHECK(cudaDeviceSynchronize());
  for (int id = 0; id < smoothMsgs; id++)
    MPI_Isend(smoothSendBuf[id],smoothCount[id],MPI_DOUBLE,smoothSendRank[id],smoothTag[id],MPI_COMM_WORLD,&smoothSendReq[id]);

  MPI_Waitall(smoothMsgs,smoothRecvReq,MPI_STATUSES_IGNORE);

  {
    constexpr dim3 block(32,4,2);
    const dim3 grid((std::max(diMax_,djMax_)-1)/block.x+1,(std::max(djMax_,dkMax_)-1)/block.y+1,3);
    ::copyFromBuf<<<grid,block>>>(diMax_,djMax_,dkMax_,smoothRecv,smoothIn);
  }
  {
    constexpr dim3 block(4,4,4);
    const dim3 grid((di_-1)/block.x+1,(dj_-1)/block.y+1,(dk_-1)/block.z+1);
    ::smooth<<<grid,block>>>(di_,dj_,dk_,diMax_,djMax_,dkMax_,smootherRadius_,smoother_,smoothIn,dPhi_);
  }

  MPI_Waitall(smoothMsgs,smoothSendReq,MPI_STATUSES_IGNORE);
  CHECK(cudaDeviceSynchronize());
}
