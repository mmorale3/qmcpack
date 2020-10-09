//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by:
// Fionn D. Malone, malone14@llnl.gov
//    Lawrence Livermore National Laboratory
////////////////////////////////////////////////////////////////////////////////

#include "catch.hpp"
#include "Configuration.h"

#undef APP_ABORT
#define APP_ABORT(x) \
  {                  \
    std::cout << x;  \
    exit(0);         \
  }

#include <vector>

#include "multi/array.hpp"
#include "multi/array_ref.hpp"

#include "Utilities/Timer.h"

#include "AFQMC/config.h"
#include "AFQMC/config.0.h"
#include "AFQMC/Numerics/ma_blas.hpp"
#include "AFQMC/Numerics/batched_operations.hpp"
#include "AFQMC/Matrix/tests/matrix_helpers.h"
#if defined(ENABLE_CUDA)
#include "AFQMC/Numerics/detail/CUDA/blas_cuda_gpu_ptr.hpp"
#include "AFQMC/Numerics/device_kernels.hpp"
#elif defined(ENABLE_HIP)
#include "AFQMC/Numerics/detail/HIP/blas_hip_gpu_ptr.hpp"
#include "AFQMC/Numerics/device_kernels.hpp"
#endif

#include "AFQMC/Utilities/test_utils.hpp"


using boost::multi::array;
using boost::multi::array_ref;
using boost::multi::iextensions;
using std::copy_n;

namespace qmcplusplus
{
using namespace afqmc;

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
template<typename T>
using Alloc = device::device_allocator<T>;
#else
template<typename T>
using Alloc = std::allocator<T>;
#endif
template<typename T>
using pointer = typename Alloc<T>::pointer;


template<typename T>
using Tensor1D = array<T, 1, Alloc<T>>;
template<typename T>
using Tensor2D = array<T, 2, Alloc<T>>;
template<typename T>
using Tensor3D = array<T, 3, Alloc<T>>;

TEST_CASE("axpyBatched", "[Numerics][misc_kernels]")
{
  // Only implemented for complex
  Alloc<std::complex<double>> alloc{};
  Tensor2D<std::complex<double>> y({3, 4}, 1.0, alloc);
  Tensor2D<std::complex<double>> x({3, 4}, 1.0, alloc);
  Tensor1D<std::complex<double>> a(iextensions<1u>{3}, 2.0, alloc);
  std::vector<pointer<std::complex<double>>> x_batched, y_batched;
  for (int i = 0; i < x.size(0); i++)
  {
    x_batched.emplace_back(x[i].origin());
    y_batched.emplace_back(y[i].origin());
  }
  using ma::axpyBatched;
  axpyBatched(x.size(1), to_address(a.origin()), x_batched.data(), 1, y_batched.data(), 1, x_batched.size());
  // 1 + 2 = 3.
  Tensor2D<std::complex<double>> ref({3, 4}, 3.0, alloc);
  verify_approx(y, ref);
}

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
// Not in dispatching routine yet, called directly from AFQMCBasePropagator.
TEST_CASE("construct_X", "[Numerics][misc_kernels]")
{
  Alloc<std::complex<double>> alloc{};
  int ncv                 = 11;
  int nsteps              = 2;
  int nwalk               = 3;
  bool fp                 = false;
  double sqrtdt           = 0.002;
  double vbound           = 40.0;
  std::complex<double> im = std::complex<double>(0.0, 1.0);
  Tensor1D<std::complex<double>> vmf(iextensions<1U>{ncv}, im, alloc);
  Tensor2D<std::complex<double>> vbias({ncv, nwalk}, 1.0, alloc);
  Tensor2D<std::complex<double>> hws({nsteps, nwalk}, -0.2, alloc);
  Tensor2D<std::complex<double>> mf({nsteps, nwalk}, 2.0, alloc);
  Tensor3D<std::complex<double>> x({ncv, nsteps, nwalk}, 0.1, alloc);
  using kernels::construct_X;
  construct_X(ncv, nsteps, nwalk, fp, sqrtdt, vbound, to_address(vmf.origin()), to_address(vbias.origin()),
              to_address(hws.origin()), to_address(mf.origin()), to_address(x.origin()));
  // captured from stdout.
  std::complex<double> ref_val = std::complex<double>(0.102, 0.08);
  Tensor3D<std::complex<double>> ref({ncv, nsteps, nwalk}, ref_val, alloc);
  verify_approx(ref, x);
}

// No cpu equivalent?
TEST_CASE("batchedDot", "[Numerics][misc_kernels]")
{
  Alloc<std::complex<double>> alloc{};
  std::complex<double> im = std::complex<double>(0.0, 1.0);
  int dim                 = 3;
  Tensor1D<std::complex<double>> y(iextensions<1U>{dim}, im, alloc);
  Tensor2D<std::complex<double>> A({dim, dim}, 1.0, alloc);
  Tensor2D<std::complex<double>> B({dim, dim}, -0.2, alloc);
  std::complex<double> alpha(2.0);
  std::complex<double> beta(-1.0);
  using kernels::batchedDot;
  batchedDot(dim, dim, alpha, to_address(A.origin()), dim, to_address(B.origin()), dim, beta, to_address(y.origin()),
             1);
  // from numpy.
  std::complex<double> ref_val(-1.2, -1.0);
  Tensor1D<std::complex<double>> ref(iextensions<1U>{dim}, ref_val, alloc);
  verify_approx(ref, y);
}


TEST_CASE("extract_overlap_matrix", "[Numerics][misc_kernels]")
{
  Alloc<ComplexType> alloc{};
  Alloc<int> ialloc{};
  int ndet = 1000;
  int nex = 5;
  int nmo = 100;
  Tensor3D<ComplexType> M({ndet, nex, nex}, ComplexType(0.0), alloc);
  Tensor2D<ComplexType> T({nmo, nmo}, ComplexType(0.0), alloc);
  boost::multi::array<ComplexType, 3> M_({ndet, nex, nex}, ComplexType(0.0));
  boost::multi::array<ComplexType, 2> T_({nmo, nmo}, ComplexType(0.0));
  Tensor1D<int> excit(iextensions<1u>{ndet*2*nex}, 0, alloc);
  std::vector<int> tmpi(ndet*2*nex);
  {
    std::vector<ComplexType> tmp(nmo*nmo);
    fillRandomMatrix(tmp);
    copy_n(tmp.data(), tmp.size(), T.origin());
    copy_n(tmp.data(), tmp.size(), T_.origin());
    fillRandomMatrix(tmpi, nmo);
    copy_n(tmpi.data(), tmpi.size(), excit.origin());
  }
  Timer timer;
  for (int i = 0; i < ndet; i++)
    for (int j = 0; j < nex; j++)
      for (int k = 0; k < nex; k++)
      {
        M_[i][j][k] = T_[tmpi[i*2*nex+j+nex]][tmpi[i*2*nex+k]];
      }
  std::cout << " Tcpu fill: " << timer.elapsed() << std::endl;
  using kernels::extract_overlap_matrix;
  timer.restart();
  extract_overlap_matrix(ndet, nex, nmo, to_address(excit.origin()), to_address(T.origin()), to_address(M.origin()));
  std::cout << " Tgpu fill: " << timer.elapsed() << std::endl;
  {
    boost::multi::array<ComplexType, 3> tmp({ndet,nex,nex}, 0.0);
    copy_n(M.data(), M.num_elements(), tmp.origin());
    for (int i = 0; i < tmp.num_elements(); i++) {
      REQUIRE(std::real(*(tmp.data()+i)) == Approx(std::real(*(M_.data()+i))));
    }
  }
}

TEST_CASE("construct_phmsd_R", "[Numerics][misc_kernels]")
{
  Alloc<ComplexType> alloc{};
  Alloc<int> ialloc{};
  int ndet = 2000;
  int nex = 7;
  int nmo = 50;
  int nelec = 15;
  int nact = 7;
  //// add term coming from identity
  boost::multi::array<ComplexType, 2> T_({nmo, nmo}, ComplexType(0.0));
  boost::multi::array<ComplexType, 3> I_({ndet, nex, nex}, ComplexType(0.0));
  boost::multi::array<ComplexType, 3> Rbuff_({ndet, nact, nelec}, ComplexType(0.0));
  boost::multi::array<ComplexType, 2> R_({nact, nelec}, ComplexType(0.0));
  boost::multi::array<ComplexType, 1> weights_(iextensions<1u>{ndet}, ComplexType(7.0));
  Tensor2D<ComplexType> T({nmo, nmo}, ComplexType(0.0), alloc);
  Tensor3D<ComplexType> I({ndet, nex, nex}, ComplexType(0.0), alloc);
  Tensor3D<ComplexType> Rbuff({ndet, nact, nelec}, ComplexType(0.0), alloc);
  Tensor2D<ComplexType> R({nact, nelec}, ComplexType(0.0), alloc);
  Tensor1D<ComplexType> weights(iextensions<1u>{ndet}, ComplexType(7.0), alloc);
  Tensor1D<int> iexcit(iextensions<1u>{ndet}, 7, ialloc);
  Tensor1D<int> orbs(iextensions<1u>{ndet}, 7, ialloc);
  {
    std::vector<ComplexType> tmp(ndet*nmo*nmo);
    fillRandomMatrix(tmp);
    copy_n(tmp.data(), T_.num_elements(), T_.origin());
    copy_n(tmp.data(), I_.num_elements(), I_.origin());
    copy_n(tmp.data(), T.num_elements(), T.origin());
    copy_n(tmp.data(), I.num_elements(), I.origin());
  }
  Timer timer;
  for (int nd = 0; nd < ndet; nd++) {
    for (int i = 0; i < nelec; ++i)
      if (i < nact) {
        Rbuff_[nd][i][i] += weights_[nd];
      }
    for (int p = 0; p < nex; ++p)
    {
      auto Rp = Rbuff_[nd][p];
      auto Ip = I_[nd][p];
      for (int q = 0; q < nex; ++q)
      {
        auto Ipq = Ip[q];
        auto Tq  = T_[q];
        //if (nd == 0) std::cout << p << " " << q << " " <<  Rp[0] << " " << Ipq << " " << Tq[0] << std::endl;
        for (int i = 0; i < nelec; ++i) {
          Rp[i] -= weights_[nd] * Ipq * Tq[i];
          //if (nd == 0) std::cout << i << " " << p << " " << q << " " <<  Rp[i] << " " << weights[nd]*Ipq*Tq[i] << std::endl;
        }
        //if (nd == 0)
          //std::cout << p << "  " << q << " " << weights_[nd] << " " << Ipq << std::endl;
        Rp[q] += weights_[nd] * Ipq;
      }
    }
    for (int p = 0; p < nact; p++) {
      for (int q = 0; q < nelec; q++) {
        R_[p][q] += Rbuff_[nd][p][q];
      }
    }
  }
  //std::cout << " Tcpu fill: " << timer.elapsed() << std::endl;
  using kernels::construct_phmsd_R;
  timer.restart();
  construct_phmsd_R(ndet, nex, nact, nelec, nmo,
                    to_address(iexcit.origin()), to_address(orbs.origin()),
                    to_address(weights.origin()), to_address(T.origin()),
                    to_address(I.origin()), to_address(Rbuff.origin()),
                    to_address(R.origin()));
  //std::cout << " Tgpu fill: " << timer.elapsed() << std::endl;
  using kernels::reduce_phmsd_R;
  timer.restart();
  reduce_phmsd_R(ndet, nex, nact, nelec, nmo,
                 to_address(iexcit.origin()), to_address(orbs.origin()),
                 to_address(weights.origin()), to_address(T.origin()),
                 to_address(I.origin()), to_address(Rbuff.origin()),
                 to_address(R.origin()));
  //std::cout << " Tgpu reduce: " << timer.elapsed() << std::endl;
  {
    boost::multi::array<ComplexType, 3> tmp({ndet,nact,nelec}, 0.0);
    copy_n(Rbuff.data(), Rbuff.num_elements(), tmp.origin());
    //std::cout << Rbuff[0][0][0] << " " <<  Rbuff[1][0][0] << std::endl;
    for (int idet = 0; idet < ndet; idet++) {
      for (int iv = 0; iv < nact; iv++) {
        for (int iel = 0; iel < nelec; iel++) {
          int i = idet * nact * nelec + iv * nelec + iel;
          //std::cout << idet << " " << iv << " " << iel << " " << std::real(*(tmp.data()+i)) << " " << std::real(*(R_.data()+i)) << std::endl;
          REQUIRE(std::real(*(tmp.data()+i)) == Approx(std::real(*(Rbuff_.data()+i))));
        }
      }
    }
    copy_n(R.data(), R.num_elements(), tmp.origin());
    //std::cout << R[0][0] << std::endl;
    for (int iv = 0; iv < nact; iv++) {
      for (int iel = 0; iel < nelec; iel++) {
        int i = iv * nelec + iel;
        //std::cout << iv << " " << iel << " " << std::real(*(tmp.data()+i)) << " " << std::real(*(R_.data()+i)) << std::endl;
        REQUIRE(std::real(*(tmp.data()+i)) == Approx(std::real(*(R_.data()+i))));
      }
    }
  }
}

#endif

} // namespace qmcplusplus
