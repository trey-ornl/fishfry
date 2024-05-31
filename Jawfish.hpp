#pragma once

#include "gpu.hpp"
#include "TimeStamp.hpp"

#include <mpi.h>

class Jawfish {
  public:
    static constexpr int lapNeighbors_ = 6;
    struct LapBuffers { double *__restrict__ buf[lapNeighbors_]; };

    /**
     * @param[in] n[3] { Global number of cells in each dimension, without ghost cells. }
     * @param[in] lo[3] { Physical location of the global lower bound of each dimension. }
     * @param[in] hi[3] { Physical location of the global upper bound of each dimension, minus one grid cell.
     *                     The one-cell difference is because of the periodic domain.
     *                     See @ref Potential_Paris_3D::Initialize for an example computation of these arguments. }
     * @param[in] m[3] { Number of MPI tasks in each dimension. }
     * @param[in] id[3] { Coordinates of this MPI task, starting at `{0,0,0}`. }
     */
    Jawfish(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]);

    ~Jawfish();

    /**
     * @return { Number of bytes needed for array arguments for @ref solve. }
     */
    size_t bytes() const { return bytes_; }

    /**
     * @detail { Solves the Poisson equation for the potential derived from the provided density.
     *           Assumes periodic boundary conditions.
     *           Assumes fields have no ghost cells.
     *           Uses a 3D FFT provided by the @ref Henry class. }
     * @param[in] bytes { Number of bytes allocated for arguments @ref density and @ref potential.
     *                    Used to ensure that the arrays have enough extra work space. }
     * @param[in,out] density { Input density field. Modified as a work array.
     *                          Must be at least @ref bytes() bytes, likely larger than the original field. }
     * @param[out] potential { Output potential. Modified as a work array.
     *                         Must be at least @ref bytes() bytes, likely larger than the actual output field. }
     */
    void solve(size_t bytes, double *density, double *potential, std::vector<TimeStamp> &stamps);

  private:
    size_t bytes_; //!< Max bytes needed for argument arrays
    double denomX_,denomY_,denomZ_;
    double *dhtMy_,*dhtSum_;
    int di_,dj_,dk_; //!< Max number of local points in each dimension
    double *dPhi_,*dRho_;
    double *dstA_,*dstB_;
    double dx_,dy_,dz_;
    double eps_;
    cudaEvent_t event_;
    cublasHandle_t handle_; //!< For GPU Blas calls
    int idi_,idj_,idk_; //!< MPI coordinates of 3D block
    int lapCount_[lapNeighbors_];
    int lapRank_[lapNeighbors_];
    int lapRecvTag_[lapNeighbors_];
    int lapSendTag_[lapNeighbors_];
    LapBuffers lapRecv_,lapSend_;
    MPI_Request lapRecvReq_[lapNeighbors_],lapSendReq_[lapNeighbors_];
    int maxIters_,minIters_;
    int mi_,mj_,mk_; //!< Number of MPI tasks in each dimension of 3D domain
    int ni_,nj_,nk_; //!< Global number of real points in each dimension
    int neighborIters_;
    cufftHandle planI_,planJ_,planK_; //!< For D2Z FFTs used by local 3D DST
    double scaleDST_;
    double scaleI_,scaleJ_,scaleK_; //!< Frequency-independent terms in Poisson solve
    double si_,sj_,sk_;
    double *smoother_;
    int smootherIters_;
    int smootherRadius_;

    void addLocalDST();
    void diffLaplacian();
    void initSmoother();
    void setLaplacian(const double *in, double *out);
};
