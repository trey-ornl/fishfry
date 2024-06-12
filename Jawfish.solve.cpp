#include "Jawfish.hpp"

void Jawfish::solve(const size_t bytes, double *const density, double *const potential, std::vector<TimeStamp> &stamps)
{
  stamps.clear();
  stamps.push_back({"Total",MPI_Wtime()});

  assert(bytes >= bytes_);
  const int n = di_*dj_*dk_;
  double mySum = 0;
  CHECK(cublasDasum(handle_,n,density,1,&mySum));
  CHECK(cudaEventRecord(event_,0));
  //CHECK(cudaMemset(potential,0,bytes_));
  CHECK(cudaMemset(dPhi_,0,bytes_));
  CHECK(cudaMemcpyAsync(dRho_,density,bytes_,cudaMemcpyDeviceToDevice,0));

  CHECK(cudaEventSynchronize(event_));
  double inSum = 0;
  MPI_Allreduce(&mySum,&inSum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  const bool root = (idi_+idj_+idk_ == 0);
  if (root) fprintf(stderr,"# density asum %g\n",inSum);
  stamps.push_back({"Input ASum",MPI_Wtime()});

  double err = 0;
  for (int outer = 0; outer < maxIters_; outer++) {
#if 0
    for (int inner = 0; inner < neighborIters_; inner++) {
      addLocalDST();
      for (int smooth = 0; smooth < smootherIters_; smooth++) {
        diffLaplacian(density);
        smoothEdges();
      }
      diffLaplacian(density);
    }
#endif
    addGlobalDHT();
    plot(potential,false);
    plot(dPhi_,true);
    diffLaplacian(density);
    if ((outer+1) >= minIters_) {
      double mySum = 0;
      CHECK(cublasDasum(handle_,n,dRho_,1,&mySum));
      CHECK(cudaDeviceSynchronize());
      double sum = 0;
      MPI_Allreduce(&mySum,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      err = sum/inSum;
      if (root) fprintf(stderr,"# iter %d density err %g\n",outer,err);
      if (err <= eps_) break;
    }
  }
  addLocalDST();
  diffLaplacian(density);
  smoothEdges();
  diffLaplacian(density);
  plot(dRho_);
  CHECK(cudaMemcpy(potential,dPhi_,bytes_,cudaMemcpyDeviceToDevice));
}
