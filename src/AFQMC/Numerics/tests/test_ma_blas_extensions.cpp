//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Fionn D. Malone
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

#include "Utilities/Timer.h"

#include "AFQMC/config.h"
#include "AFQMC/config.0.h"
#include "AFQMC/Numerics/ma_operations.hpp"
#include "AFQMC/Numerics/ma_blas.hpp"
#include "AFQMC/Numerics/ma_small_mat_ops.hpp"
#include "AFQMC/Matrix/tests/matrix_helpers.h"
#include "AFQMC/Numerics/device_kernels.hpp"
#include "AFQMC/Utilities/test_utils.hpp"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"


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
template<typename T>
using Tensor4D = array<T, 4, Alloc<T>>;
template<typename T>
using Tensor5D = array<T, 5, Alloc<T>>;

template<typename T>
using Tensor2D_ref = array_ref<T, 2, pointer<T>>;
template<typename T>
using Tensor1D_ref = array_ref<T, 1, pointer<T>>;

template<typename T>
void create_data(std::vector<T>& buffer, T scale)
{
  T count = T(0);
  for (int i = 0; i < buffer.size(); i++)
  {
    buffer[i] = count;
    count += T(1) / scale;
  }
}

TEST_CASE("adotpby", "[Numerics][ma_blas_extensions]")
{
  size_t n = 1025;
  SECTION("double")
  {
    Alloc<double> alloc{};
    Tensor1D<double> y = {0.0};
    Tensor1D<double> a(n);
    Tensor1D<double> b(n);
    double alpha = 0.5;
    double beta  = 0.0;
    for (int i = 0; i < a.size(); i++)
    {
      a[i] = 0.1;
      b[i] = 0.1;
    }
    using ma::adotpby;
    adotpby(alpha, a, b, beta, y.origin());
    REQUIRE(y[0] == Approx(0.5 * a.size() * 0.01));
  }
  SECTION("complex")
  {
    Alloc<std::complex<double>> alloc{};
    Tensor1D<std::complex<double>> y = {0.0};
    Tensor1D<std::complex<double>> a(n);
    Tensor1D<std::complex<double>> b(n);
    std::complex<double> alpha = 0.5;
    std::complex<double> beta  = 0.0;
    std::vector<std::complex<double>> y_cpu(1);
    for (int i = 0; i < a.size(); i++)
    {
      a[i] = ComplexType(0.1, 0.1);
      b[i] = ComplexType(0.1, -0.1);
    }
    using ma::adotpby;
    adotpby(alpha, a, b, beta, y.origin());
    copy_n(y.data(), y.size(), y_cpu.data());
    REQUIRE(std::real(y_cpu[0]) == Approx(a.size() * 0.01));
    REQUIRE(std::imag(y_cpu[0]) == Approx(0.00));
  }
}

TEST_CASE("axty", "[Numerics][ma_blas_extensions]")
{
  Alloc<double> alloc{};
  Tensor1D<double> y   = {1.0, 2.0, 3.0, 4.0};
  Tensor1D<double> x   = {0.1, 5.2, 88.4, 0.001};
  Tensor1D<double> ref = {3.33000e-01, 3.46320e+01, 8.83116e+02, 1.33200e-02};
  double alpha         = 3.33;
  using ma::axty;
  axty(alpha, x, y);
  verify_approx(y, ref);
}

TEST_CASE("axty2D", "[Numerics][ma_blas_extensions]")
{
  Alloc<double> alloc{};
  Tensor2D<double> y   = {{1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}};
  Tensor2D<double> x   = {{0.1, 5.2, 88.4, 0.001}, {0.1, 5.2, 88.4, 0.001}};
  Tensor2D<double> ref = {{3.33000e-01, 3.46320e+01, 8.83116e+02, 1.33200e-02},
                          {3.33000e-01, 3.46320e+01, 8.83116e+02, 1.33200e-02}};
  double alpha         = 3.33;
  using ma::axty;
  axty(alpha, x, y);
  verify_approx(y, ref);
}

TEST_CASE("acAxpbB", "[Numerics][ma_blas_extensions]")
{
  Alloc<ComplexType> alloc{};
  Tensor2D<ComplexType> z = {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};
  ComplexType val         = ComplexType(2.33, -0.24);
  Tensor2D<ComplexType> y = {{val, val, val}, {val, val, val}, {val, val, val}};
  Tensor1D<ComplexType> x = {-val, val, -val};
  ComplexType alpha       = 3.33;
  ComplexType beta        = 0.33;
  using ma::acAxpbB;
  acAxpbB(alpha, y, x, beta, z);
  ComplexType r1            = -17.940045;
  ComplexType r2            = 18.600045;
  Tensor2D<ComplexType> ref = {{r1, r2, r1}, {r1, r2, r1}, {r1, r2, r1}};
  verify_approx(z, ref);
}

TEST_CASE("adiagApy", "[Numerics][ma_blas_extensions]")
{
  Alloc<ComplexType> alloc{};
  ComplexType val         = ComplexType(2.33, -0.24);
  Tensor2D<ComplexType> A = {{val, val, val}, {val, val, val}, {val, val, val}};
  Tensor1D<ComplexType> y = {-val, val, -val};
  ComplexType alpha       = 3.33;
  using ma::adiagApy;
  adiagApy(alpha, A, y);
  ComplexType r1            = ComplexType(5.4289, -0.5592);
  ComplexType r2            = ComplexType(10.0889, -1.0392);
  Tensor1D<ComplexType> ref = {r1, r2, r1};
  verify_approx(y, ref);
}

TEST_CASE("sum", "[Numerics][ma_blas_extensions]")
{
  Alloc<double> alloc{};
  Tensor1D<double> y = {1, 2, 3};
  using ma::sum;
  double res = sum(y);
  REQUIRE(res == Approx(6.0));
}

TEST_CASE("sum2D", "[Numerics][ma_blas_extensions]")
{
  Alloc<double> alloc{};
  std::vector<double> buffer(3 * 3);
  Tensor2D<double> y({3, 3}, alloc);
  create_data(buffer, 1.0);
  copy_n(buffer.data(), buffer.size(), y.origin());
  using ma::sum;
  double res = sum(y);
  REQUIRE(res == Approx(36.0));
}

TEST_CASE("sum3D", "[Numerics][ma_blas_extensions]")
{
  Alloc<double> alloc{};
  std::vector<double> buffer(3 * 3 * 3);
  Tensor3D<double> y({3, 3, 3}, alloc);
  create_data(buffer, 1.0);
  copy_n(buffer.data(), buffer.size(), y.origin());
  using ma::sum;
  double res = sum(y);
  REQUIRE(res == Approx(351.0));
}

TEST_CASE("sum4D", "[Numerics][ma_blas_extensions]")
{
  Alloc<double> alloc{};
  std::vector<double> buffer(3 * 3 * 3 * 3);
  Tensor4D<double> y({3, 3, 3, 3}, alloc);
  create_data(buffer, 1.0);
  copy_n(buffer.data(), buffer.size(), y.origin());
  using ma::sum;
  double res = sum(y);
  REQUIRE(res == Approx(3240.0));
}

TEST_CASE("zero_complex_part", "[Numerics][ma_blas_extensions]")
{
  Alloc<ComplexType> alloc{};
  ComplexType val           = ComplexType(1.0, -1.0);
  Tensor1D<ComplexType> y   = {val, val, val};
  Tensor1D<ComplexType> res = {1.0, 1.0, 1.0};
  using ma::zero_complex_part;
  zero_complex_part(y);
  verify_approx(y, res);
}

TEST_CASE("set_identity2D", "[Numerics][ma_blas_extensions]")
{
  Alloc<double> alloc{};
  std::vector<double> buffer(3 * 3);
  Tensor2D<double> y({3, 3}, alloc);
  copy_n(buffer.data(), buffer.size(), y.origin());
  using ma::set_identity;
  set_identity(y);
  REQUIRE(y[0][0] == Approx(1.0));
  REQUIRE(y[1][1] == Approx(1.0));
  REQUIRE(y[2][2] == Approx(1.0));
}

TEST_CASE("set_identity3D", "[Numerics][ma_blas_extensions]")
{
  Alloc<double> alloc{};
  std::vector<double> buffer(3 * 3 * 3);
  Tensor3D<double> y({3, 3, 3}, alloc);
  copy_n(buffer.data(), buffer.size(), y.origin());
  using ma::set_identity;
  set_identity(y);
  REQUIRE(y[0][0][0] == Approx(1.0));
  REQUIRE(y[1][1][1] == Approx(1.0));
  REQUIRE(y[2][2][2] == Approx(1.0));
}

TEST_CASE("fill2D", "[Numerics][ma_blas_extensions]")
{
  Alloc<double> alloc{};
  std::vector<double> buffer(2 * 2);
  Tensor2D<double> y({2, 2}, alloc);
  copy_n(buffer.data(), buffer.size(), y.origin());
  using ma::fill;
  fill(y, 2.0);
  Tensor2D<double> ref = {{2.0, 2.0}, {2.0, 2.0}};
  verify_approx(y, ref);
}

TEST_CASE("get_diagonal_strided", "[Numerics][ma_blas_extensions]")
{
  Alloc<ComplexType> alloc{};
  int nk = 2;
  int ni = 3;
  int nj = 3;
  Tensor2D<ComplexType> A({nk, ni}, 0.0, alloc);
  Tensor3D<ComplexType> B({nk, ni, nj}, ComplexType(1.0, -3.0), alloc);
  B[0][0][0] = ComplexType(1.0);
  B[0][2][2] = ComplexType(0, -1.0);
  B[1][0][0] = ComplexType(1.0);
  B[1][2][2] = ComplexType(0, -1.0);
  using ma::get_diagonal_strided;
  get_diagonal_strided(B, A);
  Tensor2D<ComplexType> ref = {{ComplexType(1.0), ComplexType(1.0, -3.0), ComplexType(0, -1.0)},
                               {ComplexType(1.0), ComplexType(1.0, -3.0), ComplexType(0, -1.0)}};
  verify_approx(A, ref);
}

TEST_CASE("fill_irregular", "[Numerics][ma_blas_extensions]")
{
  Alloc<ComplexType> alloc{};
  Alloc<int> ialloc{};
  int ndet = 4;
  int nex = 2;
  int nmo = 10;
  Tensor3D<ComplexType> M({ndet, nex, nex}, ComplexType(0.0), alloc);
  Tensor2D<ComplexType> T({nmo, nmo}, ComplexType(0.0), alloc);
  Tensor1D<int> excit = {7, 9, 3, 4, 7, 9, 3, 4, 7, 9, 3, 4, 7, 9, 3, 4, 7, 9, 3, 4, 7, 9, 3, 4, 7, 9, 3, 4, 7, 9, 3, 4};
  std::vector<int> excit_ = {7, 9, 3, 4, 7, 9, 3, 4};
  boost::multi::array<ComplexType, 3>  M_({ndet, nex, nex}, ComplexType(0,0));
  boost::multi::array<ComplexType, 2>  T_({nmo, nmo}, ComplexType(0,0));
  for (int i = 0; i < nmo; i++) {
    for (int j = 0; j < nmo; j++) {
      T_[i][j] = ComplexType(j);
    }
  }
  using std::copy_n;
  copy_n(T_.data(), T_.num_elements(), T.origin());
  // CPU version
  for (int id = 0; id < ndet; id++) {
    for (int p = 0; p < nex; p++) {
      for (int q = 0; q < nex; q++) {
        M_[id][p][q] = T_[excit_[p+nex]][excit_[q]];
        //if (id == 0)
          //std::cout << excit_[p+nex] << " " << excit_[q] << " " << T_[excit_[p+nex]][excit_[q]] << std::endl ;
        REQUIRE(std::real(M_[id][p][q]) == Approx(excit_[q]));
      }
    }
  }
  using kernels::irregular_fill;
  irregular_fill(ndet, nex, nmo, to_address(excit.origin()), to_address(T.origin()), to_address(M.origin()));
  boost::multi::array<ComplexType, 3> tmp({ndet,nex,nex}, 0.0);
  copy_n(M.data(), M.num_elements(), tmp.origin());
  for (int i = 0; i < tmp.num_elements(); i++) {
    REQUIRE(std::real(*(tmp.data()+i)) == Approx(std::real(*(M_.data()+i))));
  }
}

TEST_CASE("calculate_overlaps", "[Numerics][ma_blas_extensions]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node  = world.split_shared(world.rank());
  arch::INIT(node);
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
  Tensor1D<int> IWORK(iextensions<1u>{ndet*(nex+1)}, alloc);
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
  using kernels::irregular_fill;
  timer.restart();
  irregular_fill(ndet, nex, nmo, to_address(excit.origin()), to_address(T.origin()), to_address(M.origin()));
  std::cout << " Tgpu fill: " << timer.elapsed() << std::endl;
  {
    boost::multi::array<ComplexType, 3> tmp({ndet,nex,nex}, 0.0);
    copy_n(M.data(), M.num_elements(), tmp.origin());
    for (int i = 0; i < tmp.num_elements(); i++) {
      REQUIRE(std::real(*(tmp.data()+i)) == Approx(std::real(*(M_.data()+i))));
    }
  }
  std::vector<pointer<ComplexType>> Marray;
  for (int i = 0; i < ndet; i++)
  {
    Marray.emplace_back(M[i].origin());
  }
  Tensor1D<ComplexType> ovlp(iextensions<1u>{ndet}, 0.0, alloc);
  using ma::getrfBatched;
  timer.restart();
  getrfBatched(nex, Marray.data(), nex, ma::pointer_dispatch(IWORK.origin()),
               ma::pointer_dispatch(IWORK.origin()) + ndet*nex, ndet);
  using ma::batched_determinant_from_getrf;
  batched_determinant_from_getrf(nex, Marray.data(), nex, IWORK.origin(), nex, ComplexType(0.0), to_address(ovlp.origin()), ndet);
  std::cout << " Tgpu det: " << timer.elapsed() << std::endl;
  std::vector<ComplexType> ovlp_host(ndet, 0.0);
  copy_n(ovlp.origin(), ovlp.num_elements(), ovlp_host.data());
  using ma::determinant;
  boost::multi::array<int, 1> pivot(iextensions<1u>{nex});
  boost::multi::array<ComplexType, 2> work({nex,nex});
  ComplexType lovlp = 0.0;
  timer.restart();
  for (int i = 0; i < ndet; i++) {
    //ComplexType det = determinant(M_[i], pivot, work, lovlp);
    //auto det = ma::D4x4(M_[i][0][0], M_[i][0][1], M_[i][1][0], M_[i][1][1]);
    auto det = ma::D4x4(M_[i][0][0], M_[i][0][1], M_[i][0][2], M_[i][0][3],
                        M_[i][1][0], M_[i][1][1], M_[i][1][2], M_[i][1][3],
                        M_[i][2][0], M_[i][2][1], M_[i][2][2], M_[i][2][3],
                        M_[i][3][0], M_[i][3][1], M_[i][3][2], M_[i][3][3]);
    //REQUIRE(std::real(ovlp_host[i]) == Approx(std::real(det)));
  }
  std::cout << " Tcpu det: " << timer.elapsed() << std::endl;

}

} // namespace qmcplusplus
