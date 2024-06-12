#include "Jawfish.hpp"

#include <algorithm>
#include <cmath>

static __device__ inline double cas(const double x) { return cos(x-0.25*M_PI); }

static __global__ void dhtAdd(
  const double scale,
  const int hi, const int hj, const int hk,
  const int di, const int dj, const int dk,
  const int oi, const int oj, const int ok,
  const int ni, const int nj, const int nk,
  const double angleI, const double angleJ, const double angleK,
  const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i < di) && (j < dj) && (k < dk)) {

    const double idi = angleI*double(i+oi-ni/2);
    const double jdj = angleJ*double(j+oj-nj/2);
    const double kdk = angleK*double(k+ok-nk/2);

    double dht = 0;
    for (int kh = 0; kh < hk; kh++) {
      const double kk = kdk*double(kh-hk/2);
      for (int jh = 0; jh < hj; jh++) {
        const double jj = jdj*double(jh-hj/2);
        for (int ih = 0; ih < hi; ih++) {
          const double ii = idi*double(ih-hi/2);
          dht += in[ih+hi*(jh+hj*kh)]*cas(ii+jj+kk);
        }
      }
    }
    out[i+di*(j+dj*k)] += scale*dht;
  }
}

static __global__ void dhtForward(
  const int dh3,
  const int hi, const int hj, const int hk,
  const int bi, const int bj, const int bk,
  const int di, const int dj, const int dk,
  const int oi, const int oj, const int ok,
  const int ni, const int nj, const int nk,
  const double angleI, const double angleJ, const double angleK,
  const double *const in,
  double *const __restrict__ out)
{
  const int block = blockIdx.x%dh3;
  const int ijkh = threadIdx.x+block*blockDim.x;
  const int hij = hi*hj;
  const int kh = ijkh/hij;
  const int jh = (ijkh%hij)/hi;
  const int ih = ijkh%hi;

  if ((ih >= hi) || (jh >= hj) || (kh >= hk)) return;
  if ((ih == hi/2) && (jh == hj/2) && (kh == hk/2)) return;

  const double idh = angleI*double(ih-hi/2);
  const double jdh = angleJ*double(jh-hj/2);
  const double kdh = angleK*double(kh-hk/2);

  const int gridX = blockIdx.x/dh3;
  const int iLo = bi*gridX;
  const int iHi = std::min(iLo+bi,di);
  const int jLo = bj*blockIdx.y;
  const int jHi = std::min(jLo+bj,dj);
  const int kLo = bk*blockIdx.z;
  const int kHi = std::min(kLo+bk,dk);

  double dht = 0;
  for (int k = kLo; k < kHi; k++) {
    const double kk = kdh*double(k+ok-nk/2);
    for (int j = jLo; j < jHi; j++) {
      const double jj = jdh*double(j+oj-nj/2);
      for (int i = iLo; i < iHi; i++) {
        const double ii = idh*double(i+oi-ni/2);
        dht += in[i+di*(j+dj*k)]*cas(ii+jj+kk);
      }
    }
  }
  if (dht != 0) atomicAdd(&out[ih+hi*(jh+hj*kh)],dht);
}

static __global__ void dhtSolve(
  const int hi, const int hj, const int hk,
  const double angleI, const double angleJ, const double angleK,
  double denomX, double denomY, double denomZ,
  double *const __restrict__ inout)
{
  const int ih = threadIdx.x+blockIdx.x*blockDim.x;
  const int jh = threadIdx.y+blockIdx.y*blockDim.y;
  const int kh = threadIdx.z+blockIdx.z*blockDim.z;

  if ((ih >= hi) || (jh >= hj) || (kh >= hk)) return;
  if ((ih == hi/2) && (jh == hj/2) && (kh == hk/2)) return;

  denomX *= 2.0*(cos(angleI*double(ih-hi/2))-1.0);
  denomY *= 2.0*(cos(angleJ*double(jh-hj/2))-1.0);
  denomZ *= 2.0*(cos(angleK*double(kh-hk/2))-1.0);
  inout[ih+hi*(jh+hj*kh)] /= denomX+denomY+denomZ;
}


void Jawfish::addGlobalDHT()
{
  const int h3 = hi_*hj_*hk_;
  CHECK(cudaMemset(dhtMy_,0,h3*sizeof(double)));

  {
    const int block = std::min(1024,h3);
    const int dh3 = (h3-1)/block+1;
    
    constexpr int nb = 8; // ~(4*110)^(1/3) to fill MI250X GCD
    const dim3 grid(dh3*nb,nb,nb);
    const int bi = (di_-1)/nb+1;
    const int bj = (dj_-1)/nb+1;
    const int bk = (dk_-1)/nb+1;

    ::dhtForward<<<grid,block>>>(dh3,hi_,hj_,hk_,bi,bj,bk,di_,dj_,dk_,oi_,oj_,ok_,ni_,nj_,nk_,angleI_,angleJ_,angleK_,dRho_,dhtMy_);
  }
  CHECK(cudaDeviceSynchronize());
  MPI_Allreduce(dhtMy_,dhtSum_,h3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  
  {
    constexpr dim3 block(4,4,4);
    const dim3 grid((hi_-1)/block.x+1,(hj_-1)/block.y+1,(hk_-1)/block.z+1);
    ::dhtSolve<<<grid,block>>>(hi_,hj_,hk_,angleI_,angleJ_,angleK_,denomX_,denomY_,denomZ_,dhtSum_);
  }

  {
    const double scale = 2.0/(double(ni_)*double(nj_)*double(nk_));
    constexpr dim3 block(4,4,4);
    const dim3 grid((di_-1)/block.x+1,(dj_-1)/block.y+1,(dk_-1)/block.z+1);
    ::dhtAdd<<<grid,block>>>(scale,hi_,hj_,hk_,di_,dj_,dk_,oi_,oj_,ok_,ni_,nj_,nk_,angleI_,angleJ_,angleK_,dhtSum_,dPhi_);
  }
}

