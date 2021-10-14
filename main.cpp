#include "FishFry.hpp"
#include "PoissonPeriodic3x1DBlockedGPU.hpp"

int main(int argc, char **argv)
{
  MPI_Init(&argc,&argv);
  int rank = MPI_PROC_NULL;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  
  int nTasks[] = {0,0,0};
  int nPoints[] = {0,0,0};
  int nIters = 1;
  if (rank == 0) {
    assert(argc > 6);
    for (int i = 0; i < 3; i++) sscanf(argv[i+1],"%d",nTasks+i);
    assert((nTasks[0] > 0) && (nTasks[1] > 0) && (nTasks[2] > 0));
    assert(nTasks[0]*nTasks[1]*nTasks[2] == size);
    for (int i = 0; i < 3; i++) sscanf(argv[i+4],"%d",nPoints+i);
    assert((nPoints[0] >= nTasks[0]) && (nPoints[1] >= nTasks[1]) && (nPoints[2] >= nTasks[2]));
    if (argc > 7) sscanf(argv[7],"%d",&nIters);
  }
  MPI_Bcast(nTasks,3,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(nPoints,3,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&nIters,1,MPI_INT,0,MPI_COMM_WORLD);

  {
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&local);
    int lrank = MPI_PROC_NULL;
    MPI_Comm_rank(local,&lrank);
    int nd = 0;
    CHECK(hipGetDeviceCount(&nd));
    const int target = lrank%nd;
    CHECK(hipSetDevice(target));
    int myd = -1;
    CHECK(hipGetDevice(&myd));
    MPI_Barrier(MPI_COMM_WORLD);
    printf("# Task %d with node rank %d using device %d (%d devices per node)\n",rank,lrank,myd,nd);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  FishFry<PoissonPeriodic3x1DBlockedGPU>(nTasks,nPoints).run(nIters);
  MPI_Finalize();
  return 0;
}

