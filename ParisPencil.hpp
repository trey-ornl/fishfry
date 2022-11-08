#pragma once

#include <mpi.h>

#include "gpu.hpp"
#include "TimeStamp.hpp"

/**
 * @brief Periodic Poisson solver using @ref Henry FFT filter.
 */
class ParisPencil {
  public:

    /**
     * @param[in] n[3] { Global number of cells in each dimension, without ghost cells. }
     * @param[in] lo[3] { Physical location of the global lower bound of each dimension. }
     * @param[in] hi[3] { Physical location of the global upper bound of each dimension, minus one grid cell.
     *                     The one-cell difference is because of the periodic domain.
     *                     See @ref Potential_Paris_3D::Initialize for an example computation of these arguments. }
     * @param[in] m[3] { Number of MPI tasks in each dimension. }
     * @param[in] id[3] { Coordinates of this MPI task, starting at `{0,0,0}`. }
     */
    ParisPencil(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]);

    ~ParisPencil();

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
    void solve(size_t bytes, double *density, double *potential, std::vector<TimeStamp> &stamps) const;

  private:
    size_t bytes_; //!< Max bytes needed for argument arrays
    cufftHandle c2ci_,c2cj_,c2rk_,r2ck_; //!< Objects for forward and inverse FFTs
    MPI_Comm commI_,commJ_,commK_; //!< Communicators of fellow tasks in X, Y, and Z pencils
    double ddi_,ddj_,ddk_; //!< Frequency-independent terms in Poisson solve
    int dh_,di_,dj_,dk_; //!< Max number of local points in each dimension
    int dhq_,dip_,djp_,djq_; //!< Max number of local points in dimensions of 2D decompositions
    int idi_,idj_,idk_; //!< MPI coordinates of 3D block
    int idp_,idq_; //!< X and Y task IDs within Z pencil
    int mi_,mj_,mk_; //!< Number of MPI tasks in each dimension of 3D domain
    int mp_,mq_; //!< Number of MPI tasks in X and Y dimensions of Z pencil
    int nh_; //!< Global number of complex values in Z dimension, after R2C transform
    int ni_,nj_,nk_; //!< Global number of real points in each dimension
};
