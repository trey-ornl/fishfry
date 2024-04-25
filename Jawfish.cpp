#include "Jawfish.hpp"

Jawfish::Jawfish(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]):
  dhtMy_(nullptr),
  dhtSum_(nullptr),
  dPhi_(nullptr),
  dRho_(nullptr),
  dstA_(nullptr),
  dstB_(nullptr),
  dx_((hi[0]-lo[0])/double(n[0]-1)),
  dy_((hi[1]-lo[1])/double(n[1]-1)),
  dz_((hi[2]-lo[2])/double(n[2]-1)),
  idi_(id[0]),
  idj_(id[1]),
  idk_(id[2]),
  mi_(m[0]),
  mj_(m[1]),
  mk_(m[2]),
  ni_(n[0]),
  nj_(n[1]),
  nk_(n[2]),
  smoother_(nullptr),
  smootherIters_(1),
  smootherRadius_(0)
{
  const int di = (ni_-1)/mi_+1;
  di_ = std::min(di,ni_-idi_*di);
  const int dj = (nj_-1)/mj_+1;
  dj_ = std::min(dj,nj_-idj_*dj);
  const int dk = (nk_-1)/mk_+1;
  dk_ = std::min(dk,nk_-idk_*dk);
  bytes_ = long(di_)*long(dj_)*long(dk_)*sizeof(double);

  CHECK(cublasCreate(&handle_));

  const long dhtBytes = (mi_+1)*(mj_+1)*(mk_+1)*sizeof(double);
  CHECK(cudaMalloc(&dhtMy_,dhtBytes));
  CHECK(cudaMalloc(&dhtSum_,dhtBytes));

  CHECK(cudaMalloc(&dPhi_,bytes_));
  CHECK(cudaMalloc(&dRho_,bytes_));

  const long dstBytes = 
    2L*sizeof(double)*std::max({
      long(di_+1)*long(dj_)*long(dk_),
      long(di_)*long(dj_+1)*long(dk_),
      long(di_)*long(dj_)*long(dk_+1)});
  CHECK(cudaMalloc(&dstA_,dstBytes));
  CHECK(cudaMalloc(&dstB_,dstBytes));

  int di2 = di_+di_;
  CHECK(cufftPlanMany(&planI_,1,&di2,0,1,di2,0,1,di2+2,CUFFT_D2Z,dj_*dk_));
  int dj2 = dj_+dj_;
  CHECK(cufftPlanMany(&planJ_,1,&dj2,0,1,dj2,0,1,dj2+2,CUFFT_D2Z,di_*dk_));
  int dk2 = dk_+dk_;
  CHECK(cufftPlanMany(&planK_,1,&dk2,0,1,dk2,0,1,dk2+2,CUFFT_D2Z,di_*dj_));

  initSmoother();
}

Jawfish::~Jawfish()
{
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaFree(smoother_));
  smoother_ = nullptr;

  CHECK(cufftDestroy(planK_));
  CHECK(cufftDestroy(planJ_));
  CHECK(cufftDestroy(planI_));
  CHECK(cudaFree(dstB_));
  dstB_ = nullptr;
  CHECK(cudaFree(dstA_));
  dstA_ = nullptr;

  CHECK(cudaFree(dRho_));
  dRho_ = nullptr;
  CHECK(cudaFree(dPhi_));
  dPhi_ = nullptr;

  CHECK(cudaFree(dhtSum_));
  dhtSum_ = nullptr;
  CHECK(cudaFree(dhtMy_));
  dhtMy_ = nullptr;

  CHECK(cublasDestroy(handle_));
}
