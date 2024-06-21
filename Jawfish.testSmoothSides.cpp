#include "Jawfish.hpp"

static __global__ void init(
  const int di, const int dj, const int dk,
  const int oi, const int oj, const int ok,
  const int ni, const int nj, const int nk,
  const int radius, const double *const smoother,
  double *const __restrict__ goal,
  double *const __restrict__ rho)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= di) || (j >= dj) || (k >= dk)) return;

  const int ns = radius+1;
  double sum = 0;
  for (int kr = -radius; kr <= radius; kr++) {
    const int ks = std::abs(kr);
    const int kl = k+kr;
    const bool sideK = (kl+1 == 0) || (kl+1 == dk);
    const int kg = (k+kr+ok+nk)%nk;
    for (int jr = -radius; jr <= radius; jr++) {
      const int js = std::abs(jr);
      const int jl = j+jr;
      const bool sideJ = (jl+1 == 0) || (jl+1 == dj);
      const int jg = (j+jr+oj+nj)%nj;
      for (int ir = -radius; ir <= radius; ir++) {
        const int is = std::abs(ir);
        const int il = i+ir;
        const bool sideI = (il+1 == 0) || (il+1 == di);
        const int ig = (i+ir+oi+ni)%ni;
        if (sideI || sideJ || sideK) {
          const double in = double(1+ig+ni*(jg+nj*kg));
          assert(in);
          sum += in*smoother[is+ns*(js+ns*ks)];
        }
      }
    }
  }
  goal[i+di*(j+dj*k)] = sum;

  double rhoijk = 0;
  if ((i+1 == di) || (j+1 == dj) || (k+1 == dk)) {
    const int ig = i+oi;
    const int jg = j+oj;
    const int kg = k+ok;
    rhoijk = double(1+ig+ni*(jg+nj*kg));
    assert(rhoijk);
  }
  rho[i+di*(j+dj*k)] = rhoijk;
}

void Jawfish::testSmoothSides()
{
  double *goal = nullptr;
  CHECK(cudaMalloc(&goal,bytes_));
  CHECK(cudaMemset(goal,0,bytes_));
  constexpr dim3 block(4,4,4);
  const dim3 grid((di_-1)/block.x+1,(dj_-1)/block.y+1,(dk_-1)/block.z+1);
  ::init<<<grid,block>>>(di_,dj_,dk_,oi_,oj_,ok_,ni_,nj_,nk_,smootherRadius_,smoother_,goal,dRho_);
  smoothSides();
  printMaxDiff("# testSmoothSides dPhi",goal,dPhi_);
  stop();
}
