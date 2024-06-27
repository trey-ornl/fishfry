#include "Jawfish.hpp"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdlib>

__global__ static void chop(const int n, const double *const in, double *const __restrict out)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.z+blockDim.z*blockIdx.z;

  if ((i >= n) || (j >= n) || (k >= n)) return;

  const int m = n+n;
  const int ii = i+n;
  const int jj = j+n;
  const int kk = k+n;

  const double fi = in[0+m*(jj+m*kk)];
  const double fj = in[ii+m*(0+m*kk)];
  const double fk = in[ii+m*(jj+m*0)];

  const double dii = double(ii)*double(m-ii);
  const double djj = double(jj)*double(m-jj);
  const double dkk = double(kk)*double(m-kk);

  const double dij = dii*djj;
  const double dik = dii*dkk;
  const double djk = djj*dkk;
  const double denom = dij+dik+djk;

  out[i+n*(j+n*k)] = in[ii+m*(jj+m*kk)]-(fi*djk+fj*dik+fk*dij)/denom;
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

__global__ static void diff(const int n, const double denomX, const double denomY, const double denomZ, const double *const a, double *const __restrict__ b)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.z+blockDim.z*blockIdx.z;

  const int n1 = n+1;

  if ((i < n1) && (j < n1) && (k < n1)) {

    double sum = (i || j || k) ? 0 : -1;

    if ((j < n) && (k < n)) {
      if (i > 0) sum += a[i-1+n*(j+n*k)]*denomX;
      if (i == 0) sum += a[1+n*(j+n*k)]*2*denomX;
      else if (i+1 < n) sum += a[i+1+n*(j+n*k)]*denomX;
    }
    if ((i < n) && (k < n)) {
      if (j > 0) sum += a[i+n*(j-1+n*k)]*denomY;
      if (j == 0) sum += a[i+n*(1+n*k)]*2*denomY;
      else if (j+1 < n) sum += a[i+n*(j+1+n*k)]*denomY;
    }
    if ((i < n) && (j < n)) {
      if (k > 0) sum += a[i+n*(j+n*(k-1))]*denomZ;
      if (k == 0) sum += a[i+n*(j+n*1)]*2*denomZ;
      else if (k+1 < n) sum += a[i+n*(j+n*(k+1))]*denomZ;
    }
    if ((i < n) && (j < n) && (k < n)) sum -= a[i+n*(j+n*k)]*2*(denomX+denomY+denomZ);

    b[i+n1*(j+n1*k)] = sum;
  }
}

void Jawfish::initSmoother()
{
  smootherRadius_ = std::min({di_-1,dj_-1,dk_-1});
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
  const int m = n+n;
  const long mmm = m*m*m;
  const long dBytes = mmm*sizeof(double);
  double *d = nullptr;
  CHECK(cudaMalloc(&d,dBytes));
  CHECK(cudaMemset(d,0,dBytes));

  const long middle = n+m*(n+m*n);
  const double one = 1;
  CHECK(cudaMemcpy(d+middle,&one,sizeof(one),cudaMemcpyHostToDevice));

  const int mh = m/2+1;
  const long mmmh = m*m*mh;
  cufftDoubleComplex *z = nullptr;
  const long zBytes = mmmh*sizeof(*z);
  CHECK(cudaMalloc(&z,zBytes));
  CHECK(cudaMemset(z,0,zBytes));
  cufftHandle d2z;
  CHECK(cufftPlan3d(&d2z,m,m,m,CUFFT_D2Z));
  CHECK(cufftExecD2Z(d2z,d,z));
  CHECK(cufftDestroy(d2z));

  constexpr dim3 block(8,8,8);
  const dim3 gridH((mh-1)/block.x+1,(m-1)/block.y+1,(m-1)/block.z+1);
  ::solve<<<gridH,block>>>(m,2.0/dx_,2.0/dy_,2.0/dz_,z);
  cufftHandle z2d;
  CHECK(cufftPlan3d(&z2d,m,m,m,CUFFT_Z2D));
  CHECK(cufftExecZ2D(z2d,z,d));
  CHECK(cufftDestroy(z2d));
  CHECK(cudaFree(z));
  z = nullptr;

  const double scale = 1.0/double(mmm);
  CHECK(cublasDscal(handle_,mmm,&scale,d,1));

  const int nnn = n*n*n;
  const long smootherBytes = long(nnn)*sizeof(double);
  CHECK(cudaMalloc(&smoother_,smootherBytes));
  const dim3 grid((n-1)/block.x+1,(n-1)/block.y+1,(n-1)/block.z+1);
  ::chop<<<grid,block>>>(n,d,smoother_);
  CHECK(cudaFree(d));
  d = nullptr;

  const int n1 = n+1;
  const int n1n1n1 = n1*n1*n1;
  double *b = nullptr;
  const long bBytes = long(n1n1n1)*sizeof(double);
  CHECK(cudaMalloc(&b,bBytes));
  const dim3 grid1((n1-1)/block.x+1,(n1-1)/block.y+1,(n1-1)/block.z+1);
  ::diff<<<grid1,block>>>(n,denomX_,denomY_,denomZ_,smoother_,b);
  int iMax = 0;
  CHECK(cublasIdamax(handle_,n1n1n1,b,1,&iMax)); // first index is 1
  assert(iMax > 0);
  double aSum = 0;
  CHECK(cublasDasum(handle_,n1n1n1,b,1,&aSum));
  const double avg = aSum/double(n1n1n1);
  double bMax = 0;
  CHECK(cudaMemcpy(&bMax,b+iMax-1,sizeof(bMax),cudaMemcpyDeviceToHost));
  if (idi_+idj_+idk_ == 0) {
    fprintf(stderr,"# smoother %d^3, iters %d, avg error %g, max error %g\n",n+n-1,smootherIters_,avg,std::abs(bMax));
  }
  CHECK(cudaFree(b));
  b = nullptr;
}
