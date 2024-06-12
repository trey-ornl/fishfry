#include "Jawfish.hpp"

static __global__ void addOut(const double scale, const int di, const int dj, const int dk, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i < di-1) && (j < dj-1) && (k < dk-1)) {
    // Pick out imaginary entries
    const double inijk = in[2*(i+1)+1+2*(di+1)*(j+dj*k)];
    out[i+di*(j+dj*k)] += scale*inijk;
  }
}

static __global__ void copyIn(const int di, const int dj, const int dk, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i < di) && (j < dj) && (k < dk)) {
    const int di2 = di+di;
    if (i == 0) {
      out[di2*(j+dj*k)] = 0;
      out[di+di2*(j+dj*k)] = 0;
    } else {
      const double inijk = in[i-1+di*(j+dj*k)];
      out[i+di2*(j+dj*k)] = -inijk;
      out[di2-i+di2*(j+dj*k)] = inijk;
    }
  }
}

static __global__ void solve(double denomX2, double denomY2, double denomZ2, const double si, const double sj, const double sk, const int di, const int dj, const int dk, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i < di) && (j < dj) && (k < dk)) {
    const int di2 = di+di;
    if (i == 0) {
      out[di2*(j+dj*k)] = 0;
      out[di+di2*(j+dj*k)] = 0;
    } else {
      // Pick out imaginary entries
      double inijk = in[2*i+1+2*(di+1)*(j+dj*k)];
      // Solve Poisson in frequency space
      denomX2 *= cos(si*double(i))-1.0;
      denomY2 *= cos(sj*double(j+1))-1.0;
      denomZ2 *= cos(sk*double(k+1))-1.0;
      inijk /= (denomX2+denomY2+denomZ2);
      // Pad for D2Z
      out[i+di2*(j+dj*k)] = -inijk;
      out[di2-i+di2*(j+dj*k)] = inijk;
    }
  }
}

static __global__ void transposeXY(const int di, const int dj, const int dk, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i < di) && (j < dj) && (k < dk)) {
    const int di2 = di+di;
    if (i == 0) {
      out[di2*(j+dj*k)] = 0;
      out[di+di2*(j+dj*k)] = 0;
    } else {
      // Pick out imaginary entries and transpose
      const double injik = in[2*(j+1)+1+2*(dj+1)*(i-1+di*k)];
      out[i+di2*(j+dj*k)] = -injik;
      out[di2-i+di2*(j+dj*k)] = injik;
    }
  }
}

static __global__ void transposeXZ(const int di, const int dj, const int dk, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i < di) && (j < dj) && (k < dk)) {
    const int di2 = di+di;
    if (i == 0) {
      out[di2*(j+dj*k)] = 0;
      out[di+di2*(j+dj*k)] = 0;
    } else {
      // Pick out imaginary entries and transpose
      const double inkji = in[2*(k+1)+1+2*(dk+1)*(j+dj*(i-1))];
      out[i+di2*(j+dj*k)] = -inkji;
      out[di2-i+di2*(j+dj*k)] = inkji;
    }
  }
}

void Jawfish::addLocalDST()
{
  constexpr dim3 block{8,8,8};
  const dim3 grid{(di_-1)/block.x+1,(dj_-1)/block.y+1,(dk_-1)/block.z+1};
  ::copyIn<<<grid,block>>>(di_,dj_,dk_,dRho_,dstA_);
  CHECK(cufftExecD2Z(planI_,dstA_,(cufftDoubleComplex*)dstB_));
  ::transposeXY<<<grid,block>>>(di_,dj_,dk_,dstB_,dstA_);
  CHECK(cufftExecD2Z(planJ_,dstA_,(cufftDoubleComplex*)dstB_));
  ::transposeXZ<<<grid,block>>>(di_,dj_,dk_,dstB_,dstA_);
  CHECK(cufftExecD2Z(planK_,dstA_,(cufftDoubleComplex*)dstB_));
  ::solve<<<grid,block>>>(2*denomX_,2*denomY_,2*denomZ_,si_,sj_,sk_,di_,dj_,dk_,dstB_,dstA_);
  CHECK(cufftExecD2Z(planK_,dstA_,(cufftDoubleComplex*)dstB_));
  ::transposeXZ<<<grid,block>>>(di_,dj_,dk_,dstB_,dstA_);
  CHECK(cufftExecD2Z(planJ_,dstA_,(cufftDoubleComplex*)dstB_));
  ::transposeXY<<<grid,block>>>(di_,dj_,dk_,dstB_,dstA_);
  CHECK(cufftExecD2Z(planI_,dstA_,(cufftDoubleComplex*)dstB_));
  ::addOut<<<grid,block>>>(scaleDST_,di_,dj_,dk_,dstB_,dPhi_);
}
