#include "Jawfish.hpp"

#include <algorithm>

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

static __global__ void copyFromBuf(const int diMax, const int djMax, const int dkMax, const int mi, const int mj, const int ni, const int nj, const int nk, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= ni) || (j >= nj) || (k >= nk)) return;

  const int idi = i/diMax;
  const int idj = j/djMax;
  const int idk = k/dkMax;

  const int li = i%diMax;
  const int lj = j%djMax;
  const int lk = k%dkMax;

  const int mydi = std::min(diMax,ni-idi*diMax);
  const int mydj = std::min(djMax,nj-idj*djMax);

  const int global = (idi+mi*(idj+mj*idk))*diMax*djMax*dkMax;
  const int local = li+mydi*(lj+mydj*lk);
  out[i+ni*(j+nj*k)] = in[global+local];
}

static __global__ void copyToBuf(
  const int mydi, const int mydj, const int mydk,
  const int diMax, const int djMax,
  const double *const in, const SmoothSendBuffers send)
{
  const int p = threadIdx.x+blockIdx.x*blockDim.x;
  const int q = threadIdx.y+blockIdx.y*blockDim.y;
  const int dim = blockIdx.z;

  if ((dim == 0) && (p < mydj) && (q < mydk)) {
    send.buf[0][p+djMax*q] = in[mydi-1+mydi*(p+mydj*q)];
  }
  else if ((dim == 1) && (p < mydi) && (q < mydk)) {
    send.buf[1][p+diMax*q] = in[p+mydi*(mydj-1+mydj*q)];
  }
  else if ((dim == 2) && (p < mydi) && (q < mydj)) {
    send.buf[2][p+diMax*q] = in[p+mydi*(q+mydj*(mydk-1))];
  }
}

static __global__ void copyToBuf(const int diMax, const int djMax, const int dkMax, const int mi, const int mj, const int ni, const int nj, const int nk, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= ni) || (j >= nj) || (k >= nk)) return;

  const int idi = i/diMax;
  const int idj = j/djMax;
  const int idk = k/dkMax;

  const int li = i%diMax;
  const int lj = j%djMax;
  const int lk = k%dkMax;

  const int mydi = std::min(diMax,ni-idi*diMax);
  const int mydj = std::min(djMax,nj-idj*djMax);

  const int global = (idi+mi*(idj+mj*idk))*diMax*djMax*dkMax;
  const int local = li+mydi*(lj+mydj*lk);
  out[global+local] = in[i+ni*(j+nj*k)]; 
}

static __global__ void smooth(
  const int mydi, const int mydj, const int mydk,
  const int diMax, const int djMax, const int dkMax,
  const int radius, const double *const smoother,
  const SmoothSides in, double *const __restrict__ inout)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= mydi) || (j >= mydj) || (k >= mydk)) return;

  const int diam = radius+radius+1;
  double sum = 0;

  if (k <= radius)
    for (int rj = -radius; rj <= radius; rj++)  
      for (int ri = -radius; ri <= radius; ri++)
        sum += in.side[2][0][i+ri+diMax+3*diMax*(j+rj+djMax)]*smoother[ri+radius+diam*(rj+radius+diam*(k+radius))];

  if (k >= mydk-radius)
    for (int rj = -radius; rj <= radius; rj++)  
      for (int ri = -radius; ri <= radius; ri++)
        sum += in.side[2][1][i+ri+diMax+3*diMax*(j+rj+djMax)]*smoother[ri+radius+diam*(rj+radius+diam*(mydk-k+radius))];

  if (j <= radius)
    for (int rk = -radius; rk <= radius; rk++)  
      if ((k+rk != 0) && (k+rk != mydk-1))
        for (int ri = -radius; ri <= radius; ri++)
          sum += in.side[1][0][i+ri+diMax+3*diMax*(k+rk+dkMax)]*smoother[ri+radius+diam*(j+radius+diam*(rk+radius))];

  if (j >= mydj-radius)
    for (int rk = -radius; rk <= radius; rk++)  
      if ((k+rk != 0) && (k+rk != mydk-1))
        for (int ri = -radius; ri <= radius; ri++)
          sum += in.side[1][1][i+ri+diMax+3*diMax*(k+rk+dkMax)]*smoother[ri+radius+diam*(mydj-j+radius+diam*(rk+radius))];

  if (i <= radius)
    for (int rk = -radius; rk <= radius; rk++)  
      if ((k+rk != 0) && (k+rk != mydk-1))
        for (int rj = -radius; rj <= radius; rj++)
          if ((j+rj != 0) && (j+rj != mydj-1))
            sum += in.side[0][0][j+rj+djMax+3*djMax*(k+rk+dkMax)]*smoother[i+radius+diam*(rj+radius+diam*(rk+radius))];

  if (i >= mydi-radius)
    for (int rk = -radius; rk <= radius; rk++)  
      if ((k+rk != 0) && (k+rk != mydk-1))
        for (int rj = -radius; rj <= radius; rj++)
          if ((j+rj != 0) && (j+rj != mydj-1))
            sum += in.side[0][1][j+rj+djMax+3*djMax*(k+rk+dkMax)]*smoother[mydi-i+radius+diam*(rj+radius+diam*(rk+radius))];

  if (sum) inout[i+mydi*(j+mydj*k)] += sum;
}

static __global__ void smooth(const int diMax, const int djMax, const int dkMax, const int ni, const int nj, const int nk, const int radius, const double *const smoother, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= ni) || (j >= nj) || (k >= nk)) return;

  const int diam = radius+radius+1;
  double sum = 0;
  for (int rk = -radius; rk <= radius; rk++) {
    const int kk = (k+rk+nk)%nk;
    const bool kSide = ((kk+1)%dkMax == 0);
    for (int rj = -radius; rj <= radius; rj++) {
      const int jj = (j+rj+nj)%nj;
      const bool jSide = ((jj+1)%djMax == 0);
      for (int ri = -radius; ri <= radius; ri++) {
        const int ii = (i+ri+ni)%ni;
        const bool iSide = ((ii+1)%diMax == 0);
        if (kSide || jSide || iSide) {
          sum += in[ii+ni*(jj+nj*kk)]*smoother[ri+radius+diam*(rj+radius+diam*(rk+radius))];
        }
      }
    }
  }
  out[i+ni*(j+nj*k)] = sum;
}

void Jawfish::smoothSides()
{
  const bool root = (idi_+idj_+idk_ == 0);

  const int count = diMax_*djMax_*dkMax_;
  const long countAll = long(mi_)*long(mj_)*long(mk_)*long(count);

  double *smoothOut = nullptr;
  CHECK(cudaMalloc(&smoothOut,count*sizeof(double)));
  CHECK(cudaMemset(smoothOut,0,count*sizeof(double)));

  double *buf = nullptr;
  double *bufAll = nullptr;
  CHECK(cudaMalloc(&buf,count*sizeof(double)));
  if (root) CHECK(cudaMalloc(&bufAll,countAll*sizeof(double)));
  CHECK(cudaMemcpy(buf,dRho_,di_*dj_*dk_*sizeof(double),cudaMemcpyDeviceToDevice));
  CHECK(cudaDeviceSynchronize());
  MPI_Gather(buf,count,MPI_DOUBLE,bufAll,count,MPI_DOUBLE,0,MPI_COMM_WORLD);
  double *rho = nullptr;
  double *phi = nullptr;
  if (root) {
    const long bytes = long(ni_)*long(nj_)*long(nk_)*sizeof(double);
    CHECK(cudaMalloc(&rho,bytes));
    CHECK(cudaMalloc(&phi,bytes));
    CHECK(cudaMemset(rho,0,bytes));
    CHECK(cudaMemset(phi,0,bytes));
    const dim3 block(4,4,4);
    const dim3 grid((ni_-1)/block.x+1,(nj_-1)/block.y+1,(nk_-1)/block.z+1);
    ::copyFromBuf<<<grid,block>>>(diMax_,djMax_,dkMax_,mi_,mj_,ni_,nj_,nk_,bufAll,rho);
    ::smooth<<<grid,block>>>(diMax_,djMax_,dkMax_,ni_,nj_,nk_,smootherRadius_,smoother_,rho,phi);
    ::copyToBuf<<<grid,block>>>(diMax_,djMax_,dkMax_,mi_,mj_,ni_,nj_,nk_,phi,bufAll);
    CHECK(cudaDeviceSynchronize());
  }
  MPI_Scatter(bufAll,count,MPI_DOUBLE,buf,count,MPI_DOUBLE,0,MPI_COMM_WORLD);
  const double one = 1.0;
  CHECK(cublasDaxpy(handle_,di_*dj_*dk_,&one,buf,1,dPhi_,1));
  CHECK(cudaDeviceSynchronize());

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
    ::smooth<<<grid,block>>>(di_,dj_,dk_,diMax_,djMax_,dkMax_,smootherRadius_,smoother_,smoothIn,smoothOut);
  }

  MPI_Waitall(smoothMsgs,smoothSendReq,MPI_STATUSES_IGNORE);
  CHECK(cudaDeviceSynchronize());

  if (root) {
    double diffMax = 0;
    const int kk = dk_-1;
    for (int j = 0; j < 3*djMax_; j++) {
      const int jj = (j+djMax_)%nj_;
      for (int i = 0; i < 3*diMax_; i++) {
        const int ii = (i+diMax_)%ni_;
        const double rhoA = rho[ii+ni_*(jj+nj_*kk)];
        //const double rhoB = smoothRecv.buf[0][0][0][1][j+djMax_*k];
        const double rhoB = smoothIn.side[2][1][i+3*diMax_*j];
        const double diff = std::abs(rhoA-rhoB);
        printf("%d %d %g %g %g\n",i,j,rhoA,rhoB,diff);
        diffMax = std::max(diff,diffMax);
      }
    }
    printf("# diffMax %g\n",diffMax);
  }
  stop();

  if (root) {
    CHECK(cudaFree(phi));
    CHECK(cudaFree(rho));
    CHECK(cudaFree(bufAll));
  }
  CHECK(cudaFree(buf));
}
