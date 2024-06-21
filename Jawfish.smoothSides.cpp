#include "Jawfish.hpp"

#include <algorithm>
#include <cassert>

struct SmoothRecvBuffers { double *__restrict__ buf[3][2][3][3]; };
struct SmoothSendBuffers { double *__restrict__ buf[3][3][3]; };
struct SmoothSides { double *__restrict__ side[3][2]; };

static __global__ void copyFromBuf(
  const int di, const int dj, const int dk,
  const int radius,
  SmoothRecvBuffers recv, SmoothSides out)
{
  const int p = threadIdx.x+blockIdx.x*blockDim.x;
  const int q = threadIdx.y+blockIdx.y*blockDim.y;
  const int dim = blockIdx.z;
  const int dir = threadIdx.z;

  const int dp = (dim == 0) ? dj : di;
  const int dq = (dim == 2) ? dj: dk;

  int bp, dpr, pr;
  if (p < radius) {
    bp = 0;
    dpr = radius;
    pr = p;
  } else if (p < radius+dp) {
    bp = 1;
    dpr = dp;
    pr = p-radius;
  } else if (p < radius+dp+radius) {
    bp = 2;
    dpr = radius;
    pr = p-radius-dp;
  } else return;

  int bq, qr;
  if (q < radius) {
    bq = 0;
    qr = q;
  } else if (q < radius+dq) {
    bq = 1;
    qr = q-radius;
  } else if (q < radius+dq+radius) {
    bq = 2;
    qr = q-radius-dq;
  } else return;

  out.side[dim][dir][p+(radius+dp+radius)*q] = recv.buf[dim][dir][bq][bp][pr+dpr*qr];
}

static __global__ void copyToBuf(
  const int di, const int dj, const int dk, const int radius,
  const double *const in, const SmoothSendBuffers send)
{
  const int p = threadIdx.x+blockIdx.x*blockDim.x;
  const int q = threadIdx.y+blockIdx.y*blockDim.y;
  const int dim = blockIdx.z;

  const int dp = (dim == 0) ? dj : di;
  const int dq = (dim == 2) ? dj : dk;

  if ((p >= dp) || (q >= dq)) return;

  const double out = (dim == 0) ? in[di-1+di*(p+dj*q)] :
    (dim == 1) ? in[p+di*(dj-1+dj*q)] :
    /* dim == 2 */ in[p+di*(q+dj*(dk-1))];

  for (int bq = 0; bq < 3; bq++) {
    const int qb = bq ? q : q-dq+radius;
    const int dqb = (bq == 1) ? dq : radius;
    if ((qb < 0) || (qb >= dqb)) continue;
    for (int bp = 0; bp < 3; bp++) {
      const int pb = bp ? p : p-dp+radius;
      const int dpb = (bp == 1) ? dp : radius;
      if ((pb < 0) || (pb >= dpb)) continue;
      send.buf[dim][bq][bp][pb+dpb*qb] = out;
    }
  }
}

static __global__ void smooth(
  const int di, const int dj, const int dk,
  const int radius, const double *const smoother,
  const SmoothSides in, double *const __restrict__ inout)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= di) || (j >= dj) || (k >= dk)) return;

  const int ns = radius+1;
  double sum = 0;

  const int dirr = radius+di+radius;

  if (k < radius) {
    for (int jr = -radius; jr <= radius; jr++) {
      const int js = std::abs(jr);
      for (int ir = -radius; ir <= radius; ir++) {
        const int is = std::abs(ir);
        sum += in.side[2][0][i+ir+radius+dirr*(j+jr+radius)]*
          smoother[is+ns*(js+ns*(k+1))];
      }
    }
  }

  if (k+radius >= dk-1) {
    for (int jr = -radius; jr <= radius; jr++) {
      const int js = std::abs(jr);
      for (int ir = -radius; ir <= radius; ir++) {
        const int is = std::abs(ir);
        sum += in.side[2][1][i+ir+radius+dirr*(j+jr+radius)]*
          smoother[is+ns*(js+ns*(dk-1-k))];
      }
    }
  }

  if (j < radius) {
    for (int kr = -radius; kr <= radius; kr++) {
      if ((k+kr != -1) && (k+kr != dk-1)) {
        const int ks = std::abs(kr);
        for (int ir = -radius; ir <= radius; ir++) {
          const int is = std::abs(ir);
          sum += in.side[1][0][i+ir+radius+dirr*(k+kr+radius)]*
            smoother[is+ns*(j+1+ns*(ks))];
        }
      }
    }
  }

  if (j+radius >= dj-1) {
    for (int kr = -radius; kr <= radius; kr++) {
      if ((k+kr != -1) && (k+kr != dk-1)) {
        const int ks = std::abs(kr);
        for (int ir = -radius; ir <= radius; ir++) {
          const int is = std::abs(ir);
          sum += in.side[1][1][i+ir+radius+dirr*(k+kr+radius)]*
            smoother[is+ns*(dj-1-j+ns*(ks))];
        }
      }
    }
  }

  const int djrr = radius+dj+radius;

  if (i < radius) {
    for (int kr = -radius; kr <= radius; kr++) {
      if ((k+kr != -1) && (k+kr != dk-1)) {
        const int ks = std::abs(kr);
        for (int jr = -radius; jr <= radius; jr++) {
          if ((j+jr != -1) && (j+jr != dj-1)) {
            const int js = std::abs(jr);
            sum += in.side[0][0][j+jr+radius+djrr*(k+kr+radius)]*
              smoother[i+1+ns*(js+ns*(ks))];
          }
        }
      }
    }
  }

  if (i+radius >= di-1) {
    for (int kr = -radius; kr <= radius; kr++) {
      if ((k+kr != -1) && (k+kr != dk-1)) {
        const int ks = std::abs(kr);
        for (int jr = -radius; jr <= radius; jr++) {
          if ((j+jr != -1) && (j+jr != dj-1)) {
            const int js = std::abs(jr);
            sum += in.side[0][1][j+jr+radius+djrr*(k+kr+radius)]*
              smoother[di-1-i+ns*(js+ns*(ks))];
          }
        }
      }
    }
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

  const int radius = smootherRadius_;
  int tag = 133;
  int id = 0;

  { // X
    for (int k = -1; k < 2; k++) {
      const int countK = k ? smootherRadius_ : dk_;
      for (int j = -1; j < 2; j++) {
        const int countJ = j ? smootherRadius_ : dj_;
        const int count = countJ*countK;
        const long bytes = count*sizeof(double);
        CHECK(cudaMalloc((void**)&smoothSend.buf[0][k+1][j+1],bytes));
        CHECK(cudaMemset(smoothSend.buf[0][k+1][j+1],0,bytes));
        for (int i = -1; i < 1; i++) {
          smoothSendBuf[id] = smoothSend.buf[0][k+1][j+1];
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
    const long sideBytes = (radius+dj_+radius)*(radius+dk_+radius)*sizeof(double);
    CHECK(cudaMalloc((void**)&smoothIn.side[0][0],sideBytes));
    CHECK(cudaMalloc((void**)&smoothIn.side[0][1],sideBytes));
    CHECK(cudaMemset(smoothIn.side[0][0],0,sideBytes));
    CHECK(cudaMemset(smoothIn.side[0][1],0,sideBytes));
  }
  { // Y
    for (int k = -1; k < 2; k++) {
      const int countK = k ? smootherRadius_ : dk_;
      for (int i = -1; i < 2; i++) {
        const int countI = i ? smootherRadius_ : di_;
        const int count = countI*countK;
        const long bytes = count*sizeof(double);
        CHECK(cudaMalloc((void**)&smoothSend.buf[1][k+1][i+1],bytes));
        CHECK(cudaMemset(smoothSend.buf[1][k+1][i+1],0,bytes));
        for (int j = -1; j < 1; j++) {
          smoothSendBuf[id] = smoothSend.buf[1][k+1][i+1];
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
    const long sideBytes = (radius+di_+radius)*(radius+dk_+radius)*sizeof(double);
    CHECK(cudaMalloc((void**)&smoothIn.side[1][0],sideBytes));
    CHECK(cudaMalloc((void**)&smoothIn.side[1][1],sideBytes));
    CHECK(cudaMemset(smoothIn.side[1][0],0,sideBytes));
    CHECK(cudaMemset(smoothIn.side[1][1],0,sideBytes));
  }
  { // Z
    for (int j = -1; j < 2; j++) {
      const int countJ = j ? smootherRadius_ : dj_;
      for (int i = -1; i < 2; i++) {
        const int countI = i ? smootherRadius_ : di_;
        const int count = countI*countJ;
        const long bytes = count*sizeof(double);
        CHECK(cudaMalloc((void**)&smoothSend.buf[2][j+1][i+1],bytes));
        CHECK(cudaMemset(smoothSend.buf[2][j+1][i+1],0,bytes));
        for (int k = -1; k < 1; k++) {
          smoothSendBuf[id] = smoothSend.buf[2][j+1][i+1];
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
    const long sideBytes = (radius+di_+radius)*(radius+dj_+radius)*sizeof(double);
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
    ::copyToBuf<<<grid,block>>>(di_,dj_,dk_,smootherRadius_,dRho_,smoothSend);
  }

  CHECK(cudaDeviceSynchronize());
  for (int id = 0; id < smoothMsgs; id++)
    MPI_Isend(smoothSendBuf[id],smoothCount[id],MPI_DOUBLE,smoothSendRank[id],smoothTag[id],MPI_COMM_WORLD,&smoothSendReq[id]);

  MPI_Waitall(smoothMsgs,smoothRecvReq,MPI_STATUSES_IGNORE);

  {
    constexpr dim3 block(32,4,2);
    const int np = smootherRadius_+std::max(di_,dj_)+smootherRadius_;
    const int nq = smootherRadius_+std::max(dj_,dk_)+smootherRadius_;
    const dim3 grid((np-1)/block.x+1,(nq-1)/block.y+1,3);
    ::copyFromBuf<<<grid,block>>>(di_,dj_,dk_,smootherRadius_,smoothRecv,smoothIn);
  }
  {
    constexpr dim3 block(4,4,4);
    const dim3 grid((di_-1)/block.x+1,(dj_-1)/block.y+1,(dk_-1)/block.z+1);
    ::smooth<<<grid,block>>>(di_,dj_,dk_,smootherRadius_,smoother_,smoothIn,dPhi_);
  }

  MPI_Waitall(smoothMsgs,smoothSendReq,MPI_STATUSES_IGNORE);
  CHECK(cudaDeviceSynchronize());
}
