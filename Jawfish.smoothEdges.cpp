#include "Jawfish.hpp"

#include <algorithm>

static __global__ void copyFromBuf(const int di, const int dj, const int dk, const int mi, const int mj, const int ni, const int nj, const int nk, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= ni) || (j >= nj) || (k >= nk)) return;

  const int idi = i/di;
  const int idj = j/dj;
  const int idk = k/dk;

  const int li = i%di;
  const int lj = j%dj;
  const int lk = k%dk;

  const int mydi = std::min(di,ni-idi*di);
  const int mydj = std::min(dj,nj-idj*dj);

  const int global = (idi+mi*(idj+mj*idk))*di*dj*dk;
  const int local = li+mydi*(lj+mydj*lk);
  out[i+ni*(j+nj*k)] = in[global+local];
}

static __global__ void copyToBuf(const int di, const int dj, const int dk, const int mi, const int mj, const int ni, const int nj, const int nk, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= ni) || (j >= nj) || (k >= nk)) return;

  const int idi = i/di;
  const int idj = j/dj;
  const int idk = k/dk;

  const int li = i%di;
  const int lj = j%dj;
  const int lk = k%dk;

  const int mydi = std::min(di,ni-idi*di);
  const int mydj = std::min(dj,nj-idj*dj);

  const int global = (idi+mi*(idj+mj*idk))*di*dj*dk;
  const int local = li+mydi*(lj+mydj*lk);
  out[global+local] = in[i+ni*(j+nj*k)]; 
}

static __global__ void smooth(const int di, const int dj, const int dk, const int ni, const int nj, const int nk, const int radius, const double *const smoother, const double *const in, double *const __restrict__ out)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  const int j = threadIdx.y+blockIdx.y*blockDim.y;
  const int k = threadIdx.z+blockIdx.z*blockDim.z;

  if ((i >= ni) || (j >= nj) || (k >= nk)) return;

  const int diam = radius+radius+1;
  double sum = 0;
  for (int rk = -radius; rk <= radius; rk++) {
    const int kk = (k+rk+nk)%nk;
    const bool kEdge = ((kk+1)%dk == 0);
    for (int rj = -radius; rj <= radius; rj++) {
      const int jj = (j+rj+nj)%nj;
      const bool jEdge = ((jj+1)%dj == 0);
      for (int ri = -radius; ri <= radius; ri++) {
        const int ii = (i+ri+ni)%ni;
        const bool iEdge = ((ii+1)%di == 0);
        if (kEdge || jEdge || iEdge) {
          sum += in[ii+ni*(jj+nj*kk)]*smoother[ri+radius+diam*(rj+radius+diam*(rk+radius))];
        }
      }
    }
  }
  out[i+ni*(j+nj*k)] = sum;
}

void Jawfish::smoothEdges()
{
  const bool root = (idi_+idj_+idk_ == 0);

  const int di = (ni_-1)/mi_+1;
  const int dj = (nj_-1)/mj_+1;
  const int dk = (nk_-1)/mk_+1;
  const int count = di*dj*dk;
  const long countAll = long(mi_)*long(mj_)*long(mk_)*long(count);

  double *buf = nullptr;
  double *bufAll = nullptr;
  CHECK(cudaMalloc(&buf,count*sizeof(double)));
  if (root) CHECK(cudaMalloc(&bufAll,countAll*sizeof(double)));
  CHECK(cudaMemcpy(buf,dRho_,di_*dj_*dk_*sizeof(double),cudaMemcpyDeviceToDevice));
  CHECK(cudaDeviceSynchronize());
  MPI_Gather(buf,count,MPI_DOUBLE,bufAll,count,MPI_DOUBLE,0,MPI_COMM_WORLD);
  if (root) {
    double *rho = nullptr;
    double *phi = nullptr;
    const long bytes = long(ni_)*long(nj_)*long(nk_)*sizeof(double);
    CHECK(cudaMalloc(&rho,bytes));
    CHECK(cudaMalloc(&phi,bytes));
    CHECK(cudaMemset(rho,0,bytes));
    CHECK(cudaMemset(phi,0,bytes));
    const dim3 block(4,4,4);
    const dim3 grid((ni_-1)/block.x+1,(nj_-1)/block.y+1,(nk_-1)/block.z+1);
    ::copyFromBuf<<<grid,block>>>(di,dj,dk,mi_,mj_,ni_,nj_,nk_,bufAll,rho);
    ::smooth<<<grid,block>>>(di,dj,dk,ni_,nj_,nk_,smootherRadius_,smoother_,rho,phi);
    ::copyToBuf<<<grid,block>>>(di,dj,dk,mi_,mj_,ni_,nj_,nk_,phi,bufAll);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(phi));
    CHECK(cudaFree(rho));
  }
  MPI_Scatter(bufAll,count,MPI_DOUBLE,buf,count,MPI_DOUBLE,0,MPI_COMM_WORLD);
  const double one = 1.0;
  CHECK(cublasDaxpy(handle_,di_*dj_*dk_,&one,buf,1,dPhi_,1));
  CHECK(cudaDeviceSynchronize());

  if (root) CHECK(cudaFree(bufAll));
  CHECK(cudaFree(buf));
}
