#include "Jawfish.hpp"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdlib>

__global__ static void guess(const int n, const double denomX, const double denomY, const double denomZ, const double denomXYZ, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.z+blockDim.z*blockIdx.z;
  const int n1 = n+1;

  if ((i < n) && (j < n) && (k < n)) {

    const double in0 = in[i+n1*(j+n1*k)];

    const double inim = (i > 0) ? in[i-1+n1*(j+n1*k)] : in[1+n1*(j+n1*k)];
    const double inip = in[i+1+n1*(j+n1*k)];

    const double injm = (j > 0) ? in[i+n1*(j-1+n1*k)] : in[i+n1*(1+n1*k)];
    const double injp = in[i+n1*(j+1+n1*k)];

    const double inkm = (k > 0) ? in[i+n1*(j+n1*(k-1))] : in[i+n1*(j+n1*1)];
    const double inkp = in[i+n1*(j+n1*(k+1))];

    out[i+n*(j+n*k)] = in0; //denomXYZ*(denomX*(inim+inip)+denomY*(injm+injp)+denomZ*(inkm+inkp)-2.0*in0*(denomX+denomY+denomZ));
  }
}

__global__ static void matmul(const int n, const double denomX, const double denomY, const double denomZ, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.z+blockDim.z*blockIdx.z;
  const int n1 = n+1;

  if ((i < n1) && (j < n1) && (k < n1)) {

    const double in0 = ((i < n) && (j < n) && (k < n)) ? in[i+n*(j+n*k)] : 0;
    double sum = -2.0*in0*(denomX+denomY+denomZ);

    const double inim = ((j >= n) || (k >= n)) ? 0 : ((i > 0) ? in[i-1+n*(j+n*k)] : in[1+n*(j+n*k)]);
    const double inip = ((i+1 < n) && (j < n) && (k < n)) ? in[i+1+n*(j+n*k)] : 0;
    sum += denomX*(inim+inip);

    const double injm = ((i >= n) || (k >= n)) ? 0 : ((j > 0) ? in[i+n*(j-1+n*k)] : in[i+n*(1+n*k)]);
    const double injp = ((i < n) && (j+1 < n) && (k < n)) ? in[i+n*(j+1+n*k)] : 0;
    sum += denomY*(injm+injp);

    const double inkm = ((i >= n) || (j >= n)) ? 0 : ((k > 0) ? in[i+n*(j+n*(k-1))] : in[i+n*(j+n*1)]);
    const double inkp = ((i < n) && (j < n) && (k+1 < n)) ? in[i+n*(j+n*(k+1))] : 0;
    sum += denomZ*(inkm+inkp);

    out[i+n1*(j+n1*k)] = sum;
  }
}


__global__ static void overwrite(const int n, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.z+blockDim.z*blockIdx.z;

  if ((i < n) && (j < n) && (k < n)) {

    const int radius = n-1;
    const int nin = n+radius;
    out[i+n*(j+n*k)] = in[i+radius+nin*(j+radius+nin*(k+radius))];
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

  if (true) {
    double *smoother = nullptr;
    CHECK(cudaMalloc(&smoother,smootherBytes));
    CHECK(cudaMemset(smoother,0,smootherBytes));
    double *b = nullptr;
    CHECK(cudaMalloc(&b,smootherBytes));
    CHECK(cudaMemset(b,0,smootherBytes));
    const long rBytes = long(n1n1n1)*sizeof(double);
    double *r = nullptr;
    CHECK(cudaMalloc(&r,rBytes));
    CHECK(cudaMemset(r,0,rBytes));
    double *s = nullptr;
    CHECK(cudaMalloc(&s,rBytes));
    CHECK(cudaMemset(s,0,rBytes));
    CHECK(cudaDeviceSynchronize());

    const double denomXYZ = 1.0/(6.0*(denomX_*denomX_+denomY_*denomY_+denomZ_*denomZ_)+8.0*(denomX_*denomY_+denomX_*denomZ_+denomY_*denomZ_));

    constexpr dim3 block(4,4,4);
    const dim3 grid((n-1)/block.x+1,(n-1)/block.y+1,(n-1)/block.z+1);
    const dim3 grid1((n1-1)/block.x+1,(n1-1)/block.y+1,(n1-1)/block.z+1);
    double ssLast = 0;
    for (int iter = 0; iter < nnn; iter++) {
      ::matmul<<<grid1,block>>>(n,denomX_,denomY_,denomZ_,smoother,r);
      ::subtract<<<1,1>>>(r,1.0);
      ::guess<<<grid,block>>>(n,denomX_,denomY_,denomZ_,denomXYZ,r,b);
      ::matmul<<<grid1,block>>>(n,denomX_,denomY_,denomZ_,b,s);
      CHECK(cudaDeviceSynchronize());
      double rs = 0;
      CHECK(cublasDdot(handle_,n1n1n1,r,1,s,1,&rs));
      double ss = 0;
      CHECK(cublasDdot(handle_,n1n1n1,s,1,s,1,&ss));
      const double alpha = -rs/ss;
      CHECK(cublasDaxpy(handle_,nnn,&alpha,b,1,smoother,1));
      CHECK(cudaDeviceSynchronize());
      if (idi_+idj_+idk_ == 0) {
        printf("# %d rs %g ss %g alpha %g dss %g\n",iter,rs,ss,alpha,std::abs(ss-ssLast)/(ss+ssLast));
      }
      if (std::abs(ss-ssLast) < (ss+ssLast)*FLT_EPSILON) break;
      ssLast = ss;
    }
    ::matmul<<<grid1,block>>>(n,denomX_,denomY_,denomZ_,smoother,r);
    CHECK(cudaDeviceSynchronize());
    if (idi_+idj_+idk_ == 0) {
      for (int i = 0; i < n; i++) printf("%d %g %g\n",i,smoother[i],r[i]);
      printf("%g %g\n",2*smoother[1]*denomX_+2*smoother[n]*denomY_+2*smoother[n*n]*denomZ_-2*smoother[0]*(denomX_+denomY_+denomZ_),r[0]);
      printf("%g %g\n",(smoother[0]-2.0*smoother[1]+smoother[2])*denomX_+(2*smoother[1+n]-2*smoother[1])*denomY_+(2*smoother[1+n*n]-2*smoother[1])*denomZ_,r[1]);
    }
    stop();
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
    printf("%g\n",smoother_[0]);
  }
  CHECK(cudaFree(info));
  CHECK(cudaFree(b));
  CHECK(cudaFree(a));
  stop();
}
