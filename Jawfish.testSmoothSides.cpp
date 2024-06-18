#include "Jawfish.hpp"

static __global__ void init(
  const int di, const int dj, const int dk,
  const int diMax, const int djMax, const int dkMax,
  const int idi, const int idj, const int idk,
  const int ni, const int nj, const int nk,
  const int radius, const double *const smoother,
  double *const __restrict__ goal,
  double *const __restrict__ rho)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= di) || (j >= dj) || (k >= dk)) return;

  const int diam = radius+radius+1;
  double sum = 0;
  for (int kr = -radius; kr <= radius; kr++) {
    const int kl = k+kr;
    const bool sideK = (kl+1 == 0) || (kl+1 == dk);
    const int kg = (k+kr+idk*dkMax+nk)%nk;
    for (int jr = -radius; jr <= radius; jr++) {
      const int jl = j+jr;
      const bool sideJ = (jl+1 == 0) || (jl+1 == dj);
      const int jg = (j+jr+idj*djMax+nj)%nj;
      for (int ir = -radius; ir <= radius; ir++) {
        const int il = i+ir;
        const bool sideI = (il+1 == 0) || (il+1 == dj);
        const int ig = (i+ir+idi*diMax+ni)%ni;
        if (sideI || sideJ || sideK) {
          const double in = double(ig+ni*(jg+nj*kg));
          sum += in*smoother[ir+radius+diam*(jr+radius+diam*(kr+radius))];
          //if ((idi == 0) && (idj == 0) && (idk == 0) && (i == 0) && (j == 0) && (k == 0)) printf("init %d %d %d %g %g\n",ir,jr,kr,in,smoother[ir+radius+diam*(jr+radius+diam*(kr+radius))]);
        }
      }
    }
  }
  goal[i+di*(j+dj*k)] = sum;

  double rhoijk = 0;
  if ((i+1 == di) || (j+1 == dj) || (k+1 == dk)) {
    const int ig = i+idi*diMax;
    const int jg = j+idj*djMax;
    const int kg = k+idk*dkMax;
    rhoijk = double(ig+ni*(jg+nj*kg));
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
  ::init<<<grid,block>>>(di_,dj_,dk_,diMax_,djMax_,dkMax_,idi_,idj_,idk_,ni_,nj_,nk_,smootherRadius_,smoother_,goal,dRho_);
  smoothSides();
  printMaxDiff("# testSmoothSides dPhi",goal,dPhi_);
  stop();
}
