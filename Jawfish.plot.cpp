#include "Jawfish.hpp"

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

  const int block = (idi+mi*(idj+mj*idk))*di*dj*dk;
  const int local = li+mydi*(lj+mydj*lk);
  out[i+ni*(j+nj*k)] = in[block+local];
}

void Jawfish::plot(const double *const a) const
{
  CHECK(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

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
  CHECK(cudaMemcpy(buf,a,di_*dj_*dk_*sizeof(double),cudaMemcpyDeviceToDevice));
  CHECK(cudaDeviceSynchronize());
  MPI_Gather(buf,count,MPI_DOUBLE,bufAll,count,MPI_DOUBLE,0,MPI_COMM_WORLD);
  if (root) {
    double *ad = nullptr;
    double *ah = nullptr;
    const long bytes = long(ni_)*long(nj_)*long(nk_)*sizeof(double);
    CHECK(cudaMalloc(&ad,bytes));
    CHECK(cudaMemset(ad,0,bytes));
    const dim3 block(4,4,4);
    const dim3 grid((ni_-1)/block.x+1,(nj_-1)/block.y+1,(nk_-1)/block.z+1);
    ::copyFromBuf<<<grid,block>>>(di,dj,dk,mi_,mj_,ni_,nj_,nk_,bufAll,ad);
    CHECK(cudaHostAlloc(&ah,bytes,cudaHostAllocDefault));
    CHECK(cudaMemcpy(ah,ad,bytes,cudaMemcpyDeviceToHost));
    const int i = 2*di/3;
    for (int j = 0; j < nj_; j++) {
      for (int k = 0; k < nk_; k++) {
        const int ijk = i+ni_*(j+nj_*k);
        printf("%d %d %g\n",j,k,ah[ijk]);
      }
      printf("\n");
    }
    CHECK(cudaFreeHost(ah));
    CHECK(cudaFree(ad));
    CHECK(cudaFree(bufAll));
  }
  CHECK(cudaFree(buf));
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  exit(0);
}

