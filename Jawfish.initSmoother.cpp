#include "Jawfish.hpp"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdlib>

__global__ static void add(const int n, const double *const inA, const double *const inB, double *const __restrict out)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.z+blockDim.z*blockIdx.z;

  if ((i < n) && (j < n) && (k < n)) {
    const int m = n+n;
    const int ijkm = i+n+m*(j+n+m*(k+n));
    out[i+n*(j+n*k)] = inA[ijkm]+inB[ijkm];
  }
}

__global__ static void flatten(const int m, const double *const in, double *const __restrict out)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.z+blockDim.z*blockIdx.z;

  if ((i >= m) || (j >= m) || (k >= m)) return;

  if ((i == 0) || (j == 0) || (k == 0)) {
    out[i+m*(j+m*k)] = -in[i+m*(j+m*k)];
    return;
  }

#if 0
  if ((i == 1) || (j == 1) || (k == 1)) {
    out[i+m*(j+m*k)] = -in[i+m*(j+m*k)];
    return;
  }

  if ((i == m-1) || (j == m-1) || (k == m-1)) {
    out[i+m*(j+m*k)] = -in[i+m*(j+m*k)];
    return;
  }
#endif

#if 0
  const double fi = (in[0+m*(j+m*k)]*double(i-1)*double(i-m+1)+in[1+m*(j+m*k)]*double(i)*double(m-i))/double(m-1);
  const double fj = (in[i+m*(0+m*k)]*double(j-1)*double(j-m+1)+in[i+m*(1+m*k)]*double(j)*double(m-j))/double(m-1);
  const double fk = (in[i+m*(j+m*0)]*double(k-1)*double(k-m+1)+in[i+m*(j+m*1)]*double(k)*double(m-k))/double(m-1);
#endif
#if 1
  const double fi = in[0+m*(j+m*k)];
  const double fj = in[i+m*(0+m*k)];
  const double fk = in[i+m*(j+m*0)];

  const double dii = double(i)*double(m-i);
  const double djj = double(j)*double(m-j);
  const double dkk = double(k)*double(m-k);

#endif

#if 0
  const double scale = 10;
  const double fi = (in[0+m*(j+m*k)]*double(m-scale*i*m+scale*i*i-1)+scale*in[1+m*(j+m*k)]*double(i)*double(m-i))/double(m-1);
  const double fj = (in[i+m*(0+m*k)]*double(m-scale*j*m+scale*j*j-1)+scale*in[i+m*(1+m*k)]*double(j)*double(m-j))/double(m-1);
  const double fk = (in[i+m*(j+m*0)]*double(m-scale*k*m+scale*k*k-1)+scale*in[i+m*(j+m*1)]*double(k)*double(m-k))/double(m-1);

  const double dii = double(i)*double(m-i);
  const double djj = double(j)*double(m-j);
  const double dkk = double(k)*double(m-k);

#endif

#if 0
  const double fi = in[1+m*(j+m*k)];
  const double fj = in[i+m*(1+m*k)];
  const double fk = in[i+m*(j+m*1)];

  const double dii = double(i-1)*double(m-1-i);
  const double djj = double(j-1)*double(m-1-j);
  const double dkk = double(k-1)*double(m-1-k);
#endif

  const double dij = dii*djj;
  const double dik = dii*dkk;
  const double djk = djj*dkk;
  const double denom = dij+dik+djk;

  out[i+m*(j+m*k)] = -(fi*djk+fj*dik+fk*dij)/denom;
}

__device__ static inline double sqr(const double x) { return x*x; }

__global__ static void solve(const int m, double ddi, double ddj, double ddk, cufftDoubleComplex *const __restrict__ z)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.z+blockDim.z*blockIdx.z;

  const int mh = m/2+1;
  if ((i < mh) && (j < m) && (k < m)) {
    const double s = M_PI/double(m);
    const int ijk = i+mh*(j+m*k);
    if (i || j || k) {
      const double i2 = sqr(sin(double(i)*s)*ddi);
      const double j2 = sqr(sin(double(min(j,m-j))*s)*ddj);
      const double k2 = sqr(sin(double(min(k,m-k))*s)*ddk);
      const double d = -1.0/(i2+j2+k2);
      const cufftDoubleComplex zijk(d*z[ijk].x,d*z[ijk].y);
      z[ijk] = zijk;
    } else {
      z[ijk] = cufftDoubleComplex(0);
    }
  }
}

__global__ static void setA(const int n, double denomX, double denomY, double denomZ, double *const __restrict__ a)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.z+blockDim.z*blockIdx.z;

  const int n1 = n+1;
  
  if ((i < n1) && (j < n1) && (k < n1)) {

    if ((j < n) && (k < n)) {
      if (i > 0) a[i+n1*(j+n1*(k+n1*(i-1+n*(j+n*k))))] = denomX;
      if (i == 0) a[i+n1*(j+n1*(k+n1*(1+n*(j+n*k))))] = 2*denomX;
      else if (i+1 < n) a[i+n1*(j+n1*(k+n1*(i+1+n*(j+n*k))))] = denomX;
    }
    if ((i < n) && (k < n)) {
      if (j > 0) a[i+n1*(j+n1*(k+n1*(i+n*(j-1+n*k))))] = denomY;
      if (j == 0) a[i+n1*(j+n1*(k+n1*(i+n*(1+n*k))))] = 2*denomY;
      else if (j+1 < n) a[i+n1*(j+n1*(k+n1*(i+n*(j+1+n*k))))] = denomY;
    }
    if ((i < n) && (j < n)) {
      if (k > 0) a[i+n1*(j+n1*(k+n1*(i+n*(j+n*(k-1)))))] = denomZ;
      if (k == 0) a[i+n1*(j+n1*(k+n1*(i+n*(j+n*1))))] = 2*denomZ;
      else if (k+1 < n) a[i+n1*(j+n1*(k+n1*(i+n*(j+n*(k+1)))))] = denomZ;
    }
    if ((i < n) && (j < n) && (k < n)) a[i+n1*(j+n1*(k+n1*(i+n*(j+n*k))))] = -2*(denomX+denomY+denomZ);
  }
}

__global__ static void subtract(double *const x, const double y)
{
  *x -= y;
}

void Jawfish::initSmoother()
{
  constexpr int smootherRadiusMax = 34;
  smootherRadius_ = std::min({smootherRadiusMax,di_-1,dj_-1,dk_-1});
  const char *const jawfishSmootherRadius = getenv("JAWFISH_SMOOTHER_RADIUS");
  if (jawfishSmootherRadius) {
    sscanf(jawfishSmootherRadius,"%d",&smootherRadius_);
    assert(smootherRadius_ > 0);
  }

  const char *const jawfishSmootherIters = getenv("JAWFISH_SMOOTHER_ITERS");
  if (jawfishSmootherIters) {
    sscanf(jawfishSmootherIters,"%d",&smootherIters_);
  }
    
  const int n = smootherRadius_+1;
  const int n1 = n+1;
  const int nnn = n*n*n;
  const int n1n1n1 = n1*n1*n1;

  const long smootherBytes = long(nnn)*sizeof(double);
  CHECK(cudaMalloc(&smoother_,smootherBytes));

  double *smoother = nullptr;
  CHECK(cudaMalloc(&smoother,smootherBytes));

  if (true) {
    const int m = 2*n;
    const int mh = m/2+1;
    cufftHandle d2z, z2d;
    CHECK(cufftPlan3d(&d2z,m,m,m,CUFFT_D2Z));
    CHECK(cufftPlan3d(&z2d,m,m,m,CUFFT_Z2D));
    const long mmm = m*m*m;
    const long mmmh = m*m*mh;
    const long dBytes = mmm*sizeof(double);
    double *d = nullptr;
    CHECK(cudaMalloc(&d,dBytes));
    CHECK(cudaMemset(d,0,dBytes));
    const long middle = n+m*(n+m*n);
    const double one = 1;
    CHECK(cudaMemcpy(d+middle,&one,sizeof(one),cudaMemcpyHostToDevice));
    cufftDoubleComplex *z = nullptr;
    const long zBytes = mmmh*sizeof(*z);
    CHECK(cudaMalloc(&z,zBytes));
    CHECK(cudaMemset(z,0,zBytes));
    CHECK(cufftExecD2Z(d2z,d,z));
    constexpr dim3 block(8,8,8);
    const dim3 gridH((mh-1)/block.x+1,(m-1)/block.y+1,(m-1)/block.z+1);
    ::solve<<<gridH,block>>>(m,2.0/dx_,2.0/dy_,2.0/dz_,z);
    CHECK(cufftExecZ2D(z2d,z,d));
    const double scale = 1.0/double(mmm);
    CHECK(cublasDscal(handle_,mmm,&scale,d,1));
    double *dd = nullptr;
    CHECK(cudaMalloc(&dd,dBytes));
    CHECK(cudaMemset(dd,0,dBytes));
    const dim3 gridM((m-1)/block.x+1,(m-1)/block.y+1,(m-1)/block.z+1);
    ::flatten<<<gridM,block>>>(m,d,dd);
    const dim3 grid((n-1)/block.x+1,(n-1)/block.y+1,(n-1)/block.z+1);
    ::add<<<grid,block>>>(n,d,dd,smoother);
  }
  
  double *a = nullptr;
  const long aBytes = long(n1n1n1)*long(nnn)*sizeof(double);
  CHECK(cudaMalloc(&a,aBytes));
  CHECK(cudaMemset(a,0,aBytes));
  double *b = nullptr;
  const long bBytes = long(n1n1n1)*sizeof(double);
  CHECK(cudaMalloc(&b,bBytes));
  CHECK(cudaMemset(b,0,bBytes));

  constexpr dim3 block(8,8,8);
  const dim3 grid1((n1-1)/block.x+1,(n1-1)/block.y+1,(n1-1)/block.z+1);
  setA<<<grid1,block>>>(n,denomX_,denomY_,denomZ_,a);

  const double one = 1.0;
  CHECK(cudaMemcpy(b,&one,sizeof(one),cudaMemcpyHostToDevice));

  int infoH = 0;
  int *info = nullptr;
  CHECK(cudaMalloc(&info,sizeof(int)));
  CHECK(cudaMemset(info,0,sizeof(int)));

  CHECK(cublasDgels(handle_,CUBLAS_OP_N,n1n1n1,nnn,1,a,n1n1n1,b,n1n1n1,&infoH,info));
  assert(infoH == 0);
  CHECK(cudaMemcpy(smoother_,b,smootherBytes,cudaMemcpyDeviceToDevice));
  CHECK(cudaMemset(a,0,aBytes));
  setA<<<grid1,block>>>(n,denomX_,denomY_,denomZ_,a);
  const double alpha = 1.0;
  const double beta = 0.0;
  CHECK(cublasDgemv(handle_,CUBLAS_OP_N,n1n1n1,nnn,&alpha,a,n1n1n1,smoother_,1,&beta,b,1));
  subtract<<<1,1>>>(b,1.0);
  int iMax = -1;
  CHECK(cublasIdamax(handle_,n1n1n1,b,1,&iMax)); // first index is 1
  CHECK(cudaDeviceSynchronize());
  assert(iMax > 0);
  if (idi_+idj_+idk_ == 0) {
    double bMax = 0;
    CHECK(cudaMemcpy(&bMax,b+iMax-1,sizeof(bMax),cudaMemcpyDeviceToHost));
    fprintf(stderr,"# smoother %d^3, iters %d, max error %g\n",n+n-1,smootherIters_,bMax);
    for (int i = 0; i < n; i++) printf("%d %g %g\n",i,smoother_[i],b[i]);
    printf("%d 0 %g\n",n,b[n]);
  }
  CHECK(cublasDgemv(handle_,CUBLAS_OP_N,n1n1n1,nnn,&alpha,a,n1n1n1,smoother,1,&beta,b,1));
  subtract<<<1,1>>>(b,1.0);
  //int iMax = -1;
  CHECK(cublasIdamax(handle_,n1n1n1,b,1,&iMax)); // first index is 1
  CHECK(cudaDeviceSynchronize());
  assert(iMax > 0);
  if (idi_+idj_+idk_ == 0) {
    double bMax = 0;
    CHECK(cudaMemcpy(&bMax,b+iMax-1,sizeof(bMax),cudaMemcpyDeviceToHost));
    fprintf(stderr,"# smoother %d^3, iters %d, max error %g\n",n+n-1,smootherIters_,bMax);
    for (int i = 0; i < n; i++) printf("%d %g %g %g\n",i,smoother_[i],smoother[i],b[i]);
    printf("%d 0 0 %g\n",n,b[n]);
  }
  CHECK(cudaFree(info));
  CHECK(cudaFree(b));
  CHECK(cudaFree(a));
  stop();
}
