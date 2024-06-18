#include "Jawfish.hpp"

#include <algorithm>
#include <cstdio>

void Jawfish::printMaxDiff(
  const char *const msg,
  const double *const a, const double *const b) const
{
  double *c = nullptr;
  CHECK(cudaMalloc(&c,bytes_));
  CHECK(cudaMemcpy(c,a,bytes_,cudaMemcpyDeviceToDevice));
  const double minus1 = -1;
  const int n = di_*dj_*dk_;
  CHECK(cublasDaxpy(handle_,n,&minus1,b,1,c,1));
  int ic = -1;
  CHECK(cublasIdamax(handle_,n,c,1,&ic));
  ic--; // Fortran to C indexing
  double cMax = 0;
  CHECK(cudaMemcpy(&cMax,c+ic,sizeof(double),cudaMemcpyDeviceToHost));
  double acMax = 0;
  CHECK(cudaMemcpy(&acMax,a+ic,sizeof(double),cudaMemcpyDeviceToHost));
  double bcMax = 0;
  CHECK(cudaMemcpy(&bcMax,b+ic,sizeof(double),cudaMemcpyDeviceToHost));
  const double diff = std::abs(cMax)/std::max(std::abs(acMax),std::abs(bcMax));
  double diffMax = 0;
  MPI_Reduce(&diff,&diffMax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  int rank = MPI_PROC_NULL;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) fprintf(stderr,"%s %g\n",msg,diffMax);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  for (int r = 0; r < size; r++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == r) fprintf(stderr,"(%d,%d,%d) (%d,%d,%d) %g/max(%g,%g) = %g\n",idi_,idj_,idk_,ic%di_,(ic/di_)%dj_,ic/(di_*dj_),cMax,acMax,bcMax,diff);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  CHECK(cudaFree(c));
}

