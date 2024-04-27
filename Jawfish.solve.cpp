#include "Jawfish.hpp"
#include "gpu.hpp"

#include <mpi.h>

void Jawfish::solve(const size_t bytes, double *const density, double *const potential, std::vector<TimeStamp> &stamps)
{
  stamps.clear();
  stamps.push_back({"Total",MPI_Wtime()});

  assert(bytes >= bytes_);
  const int n = di_*dj_*dk_;
  double mySum = 0;
  CHECK(cublasDasum(handle_,n,density,1,&mySum));
  CHECK(cudaDeviceSynchronize());

  //CHECK(cudaMemset(potential,0,bytes_));
  CHECK(cudaMemcpyAsync(dRho_,density,bytes_,cudaMemcpyDeviceToDevice,0));

  double inSum = 0;
  MPI_Allreduce(&mySum,&inSum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  const bool root = (idi_+idj_+idk_ == 0);
  if (root) fprintf(stderr,"# density asum %g\n",inSum);
  stamps.push_back({"Input ASum",MPI_Wtime()});

  CHECK(cudaMemcpyAsync(dPhi_,potential,bytes_,cudaMemcpyDeviceToDevice,0));
  setLaplacian(dPhi_,dRho_);

  double alpha = -1.0;
  CHECK(cublasDaxpy(handle_,n,&alpha,density,1,dRho_,1));
  mySum = 0;
  CHECK(cublasDasum(handle_,n,dRho_,1,&mySum));
  CHECK(cudaDeviceSynchronize());
  stamps.push_back({"Laplacian Error",MPI_Wtime()});
  double errSum = 0;
  MPI_Allreduce(&mySum,&errSum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  if (root) fprintf(stderr,"# err asum %g / density asum %g = %g\n",errSum,inSum,errSum/inSum);
  return;

  double err = 0;
  for (int outer = 0; outer < maxIters_; outer++) {
    if (err <= eps_) break;
  }
}
