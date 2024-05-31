#include "Jawfish.hpp"

Jawfish::Jawfish(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]):
  dhtMy_(nullptr),
  dhtSum_(nullptr),
  dPhi_(nullptr),
  dRho_(nullptr),
  dstA_(nullptr),
  dstB_(nullptr),
  dx_((hi[2]-lo[2])/double(n[2]-1)),
  dy_((hi[1]-lo[1])/double(n[1]-1)),
  dz_((hi[0]-lo[0])/double(n[0]-1)),
  eps_(1.0e-7),
  idi_(id[2]),
  idj_(id[1]),
  idk_(id[0]),
  maxIters_(10),
  minIters_(1),
  mi_(m[2]),
  mj_(m[1]),
  mk_(m[0]),
  ni_(n[2]),
  nj_(n[1]),
  nk_(n[0]),
  neighborIters_(1),
  scaleDST_(0),
  scaleI_(2.0/(dx_*dx_)),
  scaleJ_(2.0/(dy_*dy_)),
  scaleK_(2.0/(dz_*dz_)),
  si_(0),
  sj_(0),
  sk_(0),
  smoother_(nullptr),
  smootherIters_(1),
  smootherRadius_(0)
{
  denomX_ = 1.0/(dx_*dx_);
  denomY_ = 1.0/(dy_*dy_);
  denomZ_ = 1.0/(dz_*dz_);

  const int di = (ni_-1)/mi_+1;
  di_ = std::min(di,ni_-idi_*di);
  si_ = M_PI/double(di_);
  const int dj = (nj_-1)/mj_+1;
  dj_ = std::min(dj,nj_-idj_*dj);
  sj_ = M_PI/double(dj_);
  const int dk = (nk_-1)/mk_+1;
  dk_ = std::min(dk,nk_-idk_*dk);
  sk_ = M_PI/double(dk_);
  bytes_ = long(di_)*long(dj_)*long(dk_)*sizeof(double);
  scaleDST_ = 1.0/(8.0*double(di_)*double(dj_)*double(dk_));

  CHECK(cublasCreate(&handle_));
  CHECK(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_HOST));

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

  const char *const jawfishEps = getenv("JAWFISH_EPS");
  if (jawfishEps) sscanf(jawfishEps,"%lf",&eps_);

  const char *const jawfishMaxIters = getenv("JAWFISH_MAX_OUTER_ITERS");
  if (jawfishMaxIters) sscanf(jawfishMaxIters,"%d",&maxIters_);

  const char *const jawfishMinIters = getenv("JAWFISH_MIN_OUTER_ITERS");
  if (jawfishMinIters) sscanf(jawfishMinIters,"%d",&minIters_);

  const char *const jawfishNeighborIters = getenv("JAWFISH_NEIGHBOR_ITERS");
  if (jawfishNeighborIters) sscanf(jawfishNeighborIters,"%d",&neighborIters_);

  if (idi_+idj_+idk_ == 0) fprintf(stderr,"# Jawfish eps %g, min|max outer iters %d|%d x neighbor iters %d\n",eps_,minIters_,maxIters_,neighborIters_);

  // Laplacian communication

  int lapTag = 33;

  // X
  const int xCount = dj_*dk_;
  lapCount_[0] = lapCount_[1] = xCount;
  const int idim = (idi_-1+mi_)%mi_;
  lapRank_[0] = idim+mi_*(idj_+mj_*idk_);
  const int idip = (idi_+1)%mi_;
  lapRank_[1] = idip+mi_*(idj_+mj_*idk_);
  const long xBytes = xCount*sizeof(double);
  lapRecvTag_[0] = lapSendTag_[1] = lapTag++;
  lapRecvTag_[1] = lapSendTag_[0] = lapTag++;
  CHECK(cudaMalloc((void**)&lapRecv_.buf[0],xBytes));
  CHECK(cudaMalloc((void**)&lapSend_.buf[0],xBytes));
  CHECK(cudaMalloc((void**)&lapRecv_.buf[1],xBytes));
  CHECK(cudaMalloc((void**)&lapSend_.buf[1],xBytes));

  // Y
  const int yCount = di_*dk_;
  lapCount_[2] = lapCount_[3] = yCount;
  const int idjm = (idj_-1+mj_)%mj_;
  lapRank_[2] = idi_+mi_*(idjm+mj_*idk_);
  const int idjp = (idj_+1)%mj_;
  lapRank_[3] = idi_+mi_*(idjp+mj_*idk_);
  const long yBytes = yCount*sizeof(double);
  lapRecvTag_[2] = lapSendTag_[3] = lapTag++;
  lapRecvTag_[3] = lapSendTag_[2] = lapTag++;
  CHECK(cudaMalloc((void**)&lapRecv_.buf[2],yBytes));
  CHECK(cudaMalloc((void**)&lapSend_.buf[2],yBytes));
  CHECK(cudaMalloc((void**)&lapRecv_.buf[3],yBytes));
  CHECK(cudaMalloc((void**)&lapSend_.buf[3],yBytes));

  // Z
  const int zCount = di_*dj_;
  lapCount_[4] = lapCount_[5] = di_*dj_;
  const int idkm = (idk_-1+mk_)%mk_;
  lapRank_[4] = idi_+mi_*(idj_+mj_*idkm);
  const int idkp = (idk_+1)%mk_;
  lapRank_[5] = idi_+mi_*(idj_+mj_*idkp);
  const long zBytes = zCount*sizeof(double);
  lapRecvTag_[4] = lapSendTag_[5] = lapTag++;
  lapRecvTag_[5] = lapSendTag_[4] = lapTag++;
  CHECK(cudaMalloc((void**)&lapRecv_.buf[4],zBytes));
  CHECK(cudaMalloc((void**)&lapSend_.buf[4],zBytes));
  CHECK(cudaMalloc((void**)&lapRecv_.buf[5],zBytes));
  CHECK(cudaMalloc((void**)&lapSend_.buf[5],zBytes));

  CHECK(cudaEventCreate(&event_));
}

Jawfish::~Jawfish()
{
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaEventDestroy(event_));

  for (int i = lapNeighbors_-1; i >= 0; i--) {
    CHECK(cudaFree(lapSend_.buf[i]));
    lapSend_.buf[i] = nullptr;
    CHECK(cudaFree(lapRecv_.buf[i]));
    lapRecv_.buf[i] = nullptr;
  }

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
