#include "Jawfish.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>

__global__ static void setA(const int n, const double denomX, const double denomY, const double denomZ, double *const __restrict__ a)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  const int j = threadIdx.y+blockDim.y*blockIdx.y;
  const int k = threadIdx.z+blockDim.z*blockIdx.z;
  if ((i < n) && (j < n) && (k < n)) {
    const int np2 = n+2;
    a[i+1+np2*(j+1+np2*(k+np2*(i+n*(j+n*k))))] = denomZ;
    a[i+1+np2*(j+np2*(k+1+np2*(i+n*(j+n*k))))] = denomY;
    a[i+np2*(j+1+np2*(k+1+np2*(i+n*(j+n*k))))] = denomX;
    a[i+1+np2*(j+1+np2*(k+1+np2*(i+n*(j+n*k))))] = -2.0*(denomX+denomY+denomZ);
    a[i+2+np2*(j+1+np2*(k+1+np2*(i+n*(j+n*k))))] = denomX;
    a[i+1+np2*(j+2+np2*(k+1+np2*(i+n*(j+n*k))))] = denomY;
    a[i+1+np2*(j+1+np2*(k+2+np2*(i+n*(j+n*k))))] = denomZ;
  }
}

void Jawfish::initSmoother()
{
  constexpr int smootherRadiusMax = 16;
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
    
  const int n = 2*smootherRadius_+1;
  const int np2 = n+2;
  const int n3 = n*n*n;
  const int np23 = np2*np2*np2;
  
  const long smootherBytes = long(n3)*sizeof(double);
  CHECK(cudaMalloc(&smoother_,smootherBytes));

  double *a = nullptr;
  const long aBytes = long(np23)*long(n3)*sizeof(double);
  CHECK(cudaMalloc(&a,aBytes));
  CHECK(cudaMemset(a,0,aBytes));
  double *b = nullptr;
  const long bBytes = long(np23)*sizeof(double);
  CHECK(cudaMalloc(&b,bBytes));
  CHECK(cudaMemset(b,0,bBytes));

  constexpr dim3 block(8,8,8);
  const dim3 grid((n-1)/block.x+1,(n-1)/block.y+1,(n-1)/block.z+1);
  setA<<<grid,block>>>(n,dx_,dy_,dz_,a);

  const long middle = long(smootherRadius_)+1L+long(np2)*(long(smootherRadius_)+1L+long(np2)*(long(smootherRadius_)+1L));
  CHECK(cudaDeviceSynchronize());
  b[middle] = 1.0;

  int infoH = 0;
  int *info = nullptr;
  CHECK(cudaMalloc(&info,sizeof(int)));
  CHECK(cudaMemset(info,0,sizeof(int)));

  CHECK(cublasDgels(handle_,CUBLAS_OP_N,np23,n3,1,a,np23,b,np23,&infoH,info));
  assert(infoH == 0);
  CHECK(cudaDeviceSynchronize());
  assert(*info == 0);
  CHECK(cudaMemcpyAsync(smoother_,b,smootherBytes,cudaMemcpyDeviceToDevice,0));
  CHECK(cudaMemset(a,0,aBytes));
  const double denomX = 1.0/(dx_*dx_);
  const double denomY = 1.0/(dy_*dy_);
  const double denomZ = 1.0/(dz_*dz_);
  setA<<<grid,block>>>(n,denomX,denomY,denomZ,a);
  const double alpha = 1.0;
  const double beta = 0.0;
  CHECK(cublasDgemv(handle_,CUBLAS_OP_N,np23,n3,&alpha,a,np23,smoother_,1,&beta,b,1));
  CHECK(cudaDeviceSynchronize());
  const double scale = 1.0/b[middle];
  CHECK(cublasDscal(handle_,n3,&scale,smoother_,1));
  CHECK(cublasDgemv(handle_,CUBLAS_OP_N,np23,n3,&alpha,a,np23,smoother_,1,&beta,b,1));
  CHECK(cudaMemset(&b[middle],0,sizeof(double)));
  int iMax = -1;
  CHECK(cublasIdamax(handle_,np23,b,1,&iMax)); // first index is 1
  CHECK(cudaDeviceSynchronize());
  assert(iMax > 0);

  if (idi_+idj_+idk_ == 0) fprintf(stderr,"# smoother %d^3, iters %d, max error %g\n",n,smootherIters_,b[iMax-1]);

  CHECK(cudaDeviceSynchronize());
  CHECK(cudaFree(info));
  CHECK(cudaFree(b));
  CHECK(cudaFree(a));
}
