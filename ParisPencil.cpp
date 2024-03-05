#include "ParisPencil.hpp"

#include <cmath>

__host__ __device__ static inline double sqr(const double x) { return x*x; }

ParisPencil::ParisPencil(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]):
  bytes_(0),
  ddi_{2.0*M_PI*double(n[0]-1)/(double(n[0])*(hi[0]-lo[0]))},
  ddj_{2.0*M_PI*double(n[1]-1)/(double(n[1])*(hi[1]-lo[1]))},
  ddk_{2.0*M_PI*double(n[2]-1)/(double(n[2])*(hi[2]-lo[2]))},
  idi_(id[0]),
  idj_(id[1]),
  idk_(id[2]),
  mi_(m[0]),
  mj_(m[1]),
  mk_(m[2]),
  nh_(n[2]/2+1),
  ni_(n[0]),
  nj_(n[1]),
  nk_(n[2])
{
    // Pencil sub-decomposition within a 3D block
  mq_ = int(round(sqrt(mk_)));
  while (mk_%mq_) mq_--;
  mp_ = mk_/mq_;
  assert(mp_*mq_ == mk_);

  idp_ = idk_/mq_;
  idq_ = idk_%mq_;

  // Communicators of tasks within pencils in each dimension
  {
    const int color = idi_*mj_+idj_;
    const int key = idk_;
    MPI_Comm_split(MPI_COMM_WORLD,color,key,&commK_);
  }
  {
    const int color = idi_*mp_+idp_;
    const int key = idj_*mq_+idq_;
    MPI_Comm_split(MPI_COMM_WORLD,color,key,&commJ_);
  }
  {
    const int color = idj_*mq_+idq_;
    const int key = idi_*mp_+idp_;
    MPI_Comm_split(MPI_COMM_WORLD,color,key,&commI_);
  }

  // Maximum numbers of elements for various decompositions and dimensions

  dh_ = (nh_+mk_-1)/mk_;
  di_ = (ni_+mi_-1)/mi_;
  dj_ = (nj_+mj_-1)/mj_;
  dk_ = (nk_+mk_-1)/mk_;

  dip_ = (di_+mp_-1)/mp_;
  djq_ = (dj_+mq_-1)/mq_;
  const int mjq = mj_*mq_;
  dhq_ = (nh_+mjq-1)/mjq;
  const int mip = mi_*mp_;
  djp_ = (nj_+mip-1)/mip;

  // Maximum memory needed by work arrays

  const long nMax = std::max(
    { long(di_)*long(dj_)*long(dk_),
      long(mp_)*long(mq_)*long(dip_)*long(djq_)*long(dk_),
      long(2)*long(dip_)*long(djq_)*long(mk_)*long(dh_),
      long(2)*long(dip_)*long(mp_)*long(djq_)*long(mq_)*long(dh_),
      long(2)*long(dip_)*long(djq_)*long(mjq)*long(dhq_),
      long(2)*long(dip_)*long(dhq_)*long(mip)*long(djp_),
      long(2)*djp_*long(dhq_)*long(mip)*long(dip_)
    });
  assert(nMax <= INT_MAX);
  bytes_ = nMax*sizeof(double);

  // FFT objects
  CHECK(cufftPlanMany(&c2ci_,1,&ni_,&ni_,1,ni_,&ni_,1,ni_,CUFFT_Z2Z,djp_*dhq_));
  CHECK(cufftPlanMany(&c2cj_,1,&nj_,&nj_,1,nj_,&nj_,1,nj_,CUFFT_Z2Z,dip_*dhq_));
  CHECK(cufftPlanMany(&c2rk_,1,&nk_,&nh_,1,nh_,&nk_,1,nk_,CUFFT_Z2D,dip_*djq_));
  CHECK(cufftPlanMany(&r2ck_,1,&nk_,&nk_,1,nk_,&nh_,1,nh_,CUFFT_D2Z,dip_*djq_));
}

ParisPencil::~ParisPencil()
{
  CHECK(cufftDestroy(r2ck_));
  CHECK(cufftDestroy(c2rk_));
  CHECK(cufftDestroy(c2cj_));
  CHECK(cufftDestroy(c2ci_));
  MPI_Comm_free(&commI_);
  MPI_Comm_free(&commJ_);
  MPI_Comm_free(&commK_);
}

void ParisPencil::solve(const size_t bytes, double *const density, double *const potential, std::vector<TimeStamp> &stamps) const
{
  stamps.clear();
  stamps.push_back({"Total",MPI_Wtime()});

  // Make sure arguments have enough space
  assert(bytes >= bytes_);

  double *const a = potential;
  double *const b = density;
  cufftDoubleComplex *const ac = reinterpret_cast<cufftDoubleComplex*>(a);
  cufftDoubleComplex *const bc = reinterpret_cast<cufftDoubleComplex*>(b);

  // Local copies of member variables for lambda capture

  const double ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const int di = di_, dj = dj_, dk = dk_;
  const int dhq = dhq_, dip = dip_, djp = djp_, djq = djq_;
  const int idi = idi_, idj = idj_, idk = idk_;
  const int idp = idp_, idq = idq_;
  const int mi = mi_, mj = mj_, mk = mk_;
  const int mp = mp_, mq = mq_;
  const int nh = nh_, ni = ni_, nj = nj_, nk = nk_;

  // Indices and sizes for pencil redistributions

  const int idip = idi*mp+idp;
  const int idjq = idj*mq+idq;
  const int mip = mi*mp;
  const int mjq = mj*mq;

  // Reorder 3D block into sub-pencils

  gpuFor(
    mp,mq,dip,djq,dk,
    GPU_LAMBDA(const int p, const int q, const int i, const int j, const int k) {
      const int ii = p*dip+i; 
      const int jj = q*djq+j;
      const int ia = k+dk*(j+djq*(i+dip*(q+mq*p)));
      const int ib = k+dk*(jj+dj*ii);
      a[ia] = b[ib];
    });

  // Redistribute into Z pencils

  const int countK = dip*djq*dk;
  CHECK(cudaDeviceSynchronize());
  stamps.push_back({"Reorder 3D block into sub-pencils",MPI_Wtime()});
  MPI_Alltoall(a,countK,MPI_DOUBLE,b,countK,MPI_DOUBLE,commK_);
  stamps.push_back({"Redistribute into Z pencils",MPI_Wtime()});

  // Make Z pencils contiguous in Z
  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int jLo = idj*dj+idq*djq;
    const int jHi = std::min({jLo+djq,(idj+1)*dj,nj});
    gpuFor(
      iHi-iLo,jHi-jLo,mk,dk,
      GPU_LAMBDA(const int i, const int j, const int pq, const int k) {
        const int kk = pq*dk+k;
        if (kk < nk) {
          const int ia = kk+nk*(j+djq*i);
          const int ib = k+dk*(j+djq*(i+dip*pq));
          a[ia] = b[ib];
        }
      });
  }

  // Real-to-complex FFT in Z
  CHECK(cufftExecD2Z(r2ck_,a,bc));

  // Rearrange for Y redistribution
  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int jLo = idj_*dj_+idq*djq;
    const int jHi = std::min({jLo+djq,(idj+1)*dj,nj});
    gpuFor(
      mjq,iHi-iLo,jHi-jLo,dhq,
      GPU_LAMBDA(const int q, const int i, const int j, const int k) {
        const int kk = q*dhq+k;
        if (kk < nh) {
          const int ia = k+dhq*(j+djq*(i+dip*q));
          const int ib = kk+nh*(j+djq*i);
          ac[ia] = bc[ib];
        }
      });
  }

  // Redistribute for Y pencils
  const int countJ = 2*dip*djq*dhq;
  CHECK(cudaDeviceSynchronize());
  stamps.push_back({"Real-to-complex FFT in Z",MPI_Wtime()});
  MPI_Alltoall(a,countJ,MPI_DOUBLE,b,countJ,MPI_DOUBLE,commJ_);
  stamps.push_back({"Redistribute for Y pencils",MPI_Wtime()});

  // Make Y pencils contiguous in Y
  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      kHi-kLo,iHi-iLo,mj,mq,djq,
      GPU_LAMBDA(const int k, const int i, const int r, const int q, const int j) {
        const int rdj = r*dj;
        const int jj = rdj+q*djq+j;
        if ((jj < nj) && (jj < rdj+dj)) {
          const int ia = jj+nj*(i+dip*k);
          const int ib = k+dhq*(j+djq*(i+dip*(q+mq*r)));
          ac[ia] = bc[ib];
        }
      });
  }

  // Forward FFT in Y
  CHECK(cufftExecZ2Z(c2cj_,ac,bc,CUFFT_FORWARD));

  // Rearrange for X redistribution
  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      mip,kHi-kLo,iHi-iLo,djp,
      GPU_LAMBDA(const int p, const int k, const int i, const int j) {
        const int jj = p*djp+j;
        if (jj < nj) {
          const int ia = j+djp*(i+dip*(k+dhq*p));
          const int ib = jj+nj*(i+dip*k);
          ac[ia] = bc[ib];
        }
      });
  }

  // Redistribute for X pencils
  const int countI = 2*dip*djp*dhq;
  CHECK(cudaDeviceSynchronize());
  stamps.push_back({"Forward FFT in Y",MPI_Wtime()});
  MPI_Alltoall(a,countI,MPI_DOUBLE,b,countI,MPI_DOUBLE,commI_);
  stamps.push_back({"Redistribute for X pencils",MPI_Wtime()});

  // Make X pencils contiguous in X
  {
    const int jLo = idip*djp;
    const int jHi = std::min(jLo+djp,nj);
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      jHi-jLo,kHi-kLo,mi,mp,dip,
      GPU_LAMBDA(const int j, const int k, const int r, const int p, const int i) {
        const int rdi = r*di;
        const int ii = rdi+p*dip+i;
        if ((ii < ni) && (ii < rdi+di)) {
          const int ia = ii+ni*(k+dhq*j);
          const int ib = j+djp*(i+dip*(k+dhq*(p+mp*r)));
          ac[ia] = bc[ib];
        }
      });
  }

  // Forward FFT in X
  CHECK(cufftExecZ2Z(c2ci_,ac,bc,CUFFT_FORWARD));

  // Apply filter in frequency space distributed in X pencils

  {
    const int jLo = idip*djp;
    const int jHi = std::min(jLo+djp,nj);
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);

    gpuFor(
      jHi-jLo,kHi-kLo,ni,
      GPU_LAMBDA(const int j0, const int k0, const int i) {
        const int j = jLo+j0;
        const int k = kLo+k0;
        const int iab = i+ni*(k0+dhq*j0);
        if (i || j || k) {
          const double i2 = sqr(double(min(i,ni-i))*ddi);
          const double j2 = sqr(double(min(j,nj-j))*ddj);
          const double k2 = sqr(double(k)*ddk);
          const double d = -1.0/(i2+j2+k2);
          ac[iab] = cufftDoubleComplex{d*bc[iab].x,d*bc[iab].y};
        } else {
          ac[iab] = {0,0};
        }
      });
  }

  // Backward FFT in X
  CHECK(cufftExecZ2Z(c2ci_,ac,bc,CUFFT_INVERSE));

  // Rearrange for Y redistribution
  {
    const int jLo = idip*djp;
    const int jHi = std::min(jLo+djp,nj);
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      mi,mp,jHi-jLo,kHi-kLo,dip,
      GPU_LAMBDA(const int r, const int p, const int j, const int k, const int i) {
        const int rdi = r*di;
        const int ii = rdi+p*dip+i;
        if ((ii < ni) && (ii < rdi+di)) {
          const int ia = i+dip*(k+dhq*(j+djp*(p+mp*r)));
          const int ib = ii+ni*(k+dhq*j);
          ac[ia] = bc[ib];
        }
      });
  }

  // Redistribute for Y pencils
  CHECK(cudaDeviceSynchronize());
  stamps.push_back({"X FFTs and filter",MPI_Wtime()});
  MPI_Alltoall(a,countI,MPI_DOUBLE,b,countI,MPI_DOUBLE,commI_);
  stamps.push_back({"Redistribute for Y pencils",MPI_Wtime()});

  // Make Y pencils contiguous in Y
  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      kHi-kLo,iHi-iLo,mip,djp,
      GPU_LAMBDA(const int k, const int i, const int p, const int j) {
        const int jj = p*djp+j;
        if (jj < nj) {
          const int ia = jj+nj*(i+dip*k);
          const int ib = i+dip*(k+dhq*(j+djp*p));
          ac[ia] = bc[ib];
        }
      });
  }

  // Backward FFT in Y
  CHECK(cufftExecZ2Z(c2cj_,ac,bc,CUFFT_INVERSE));

  // Rearrange for Z redistribution
  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int kLo = idjq*dhq;
    const int kHi = std::min(kLo+dhq,nh);
    gpuFor(
      mj,mq,kHi-kLo,iHi-iLo,djq,
      GPU_LAMBDA(const int r, const int q, const int k, const int i, const int j) {
        const int rdj = r*dj;
        const int jj = rdj+q*djq+j;
        if ((jj < nj) && (jj < rdj+dj)) {
          const int ia = j+djq*(i+dip*(k+dhq*(q+mq*r)));
          const int ib = jj+nj*(i+dip*k);
          ac[ia] = bc[ib];
        }
      });
  }

  // Redistribute in Z pencils
  CHECK(cudaDeviceSynchronize());
  stamps.push_back({"Backward FFT in Y",MPI_Wtime()});
  MPI_Alltoall(a,countJ,MPI_DOUBLE,b,countJ,MPI_DOUBLE,commJ_);
  stamps.push_back({"Redistribute in Z pencils",MPI_Wtime()});

  // Make Z pencils contiguous in Z
  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int jLo = idj*dj+idq*djq;
    const int jHi = std::min({jLo+djq,(idj+1)*dj,nj});
    gpuFor(
      iHi-iLo,jHi-jLo,mjq,dhq,
      GPU_LAMBDA(const int i, const int j, const int q, const int k) {
        const int kk = q*dhq+k;
        if (kk < nh) {
          const int ia = kk+nh*(j+djq*i);
          const int ib = j+djq*(i+dip*(k+dhq*q));
          ac[ia] = bc[ib];
        }
      });
  }

  // Complex-to-real FFT in Z
  CHECK(cufftExecZ2D(c2rk_,ac,b));

  // Rearrange for 3D-block redistribution
  {
    const int iLo = idi*di+idp*dip;
    const int iHi = std::min({iLo+dip,(idi+1)*di,ni});
    const int jLo = idj*dj+idq*djq;
    const int jHi = std::min({jLo+djq,(idj+1)*dj,nj});
    gpuFor(
      mk,iHi-iLo,jHi-jLo,dk,
      GPU_LAMBDA(const int pq, const int i, const int j, const int k) {
        const int kk = pq*dk+k;
        if (kk < nk) {
          const int ia = k+dk*(j+djq*(i+dip*pq));
          const int ib = kk+nk*(j+djq*i);
          a[ia] = b[ib];
        }
      });
  }

  // Redistribute for 3D blocks
  CHECK(cudaDeviceSynchronize());
  stamps.push_back({"Complex-to-real FFT in Z",MPI_Wtime()});
  MPI_Alltoall(a,countK,MPI_DOUBLE,b,countK,MPI_DOUBLE,commK_);
  stamps.push_back({"Redistribute for 3D blocks",MPI_Wtime()});

  // Rearrange into 3D blocks and apply FFT normalization
  {
    const double divN = 1.0/(double(ni)*double(nj)*double(nk));
    const int kLo = idk*dk;
    const int kHi = std::min(kLo+dk,nk);
    gpuFor(
      mp,dip,mq,djq,kHi-kLo,
      GPU_LAMBDA(const int p, const int i, const int q, const int j, const int k) {
        const int ii = p*dip+i;
        const int jj = q*djq+j;
        if ((ii < di) && (jj < dj)) {
          const int ia = k+dk*(jj+dj*ii);
          const int ib = k+dk*(j+djq*(i+dip*(q+mq*p)));
          a[ia] = divN*b[ib];
        }
      });
  }
  CHECK(cudaDeviceSynchronize());
  stamps.push_back({"Rearrange into 3D blocks and normalize",MPI_Wtime()});
}

