//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//
// File created by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory 
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_AFQMC_SLATERDETOPERATIONS_HPP
#define QMCPLUSPLUS_AFQMC_SLATERDETOPERATIONS_HPP

#include<fstream>

#include "AFQMC/config.h"
#include "AFQMC/Numerics/ma_operations.hpp"
#include "AFQMC/Numerics/csr_blas.hpp"
#include "AFQMC/SlaterDeterminantOperations/mixed_density_matrix.hpp"
#include "AFQMC/SlaterDeterminantOperations/apply_expM.hpp"

#include "mpi3/shared_communicator.hpp"
#include "AFQMC/Matrix/mpi3_SHMBuffer.hpp"
#include "type_traits/scalar_traits.h"

namespace qmcplusplus
{

namespace afqmc
{

template<class T = ComplexType> 
class SlaterDetOperations 
{
  using communicator = boost::mpi3::shared_communicator;
  using TVector = boost::multi_array<T,1>;  
  using TMatrix = boost::multi_array<T,2>;  
  using SHM_Buffer = mpi3_SHMBuffer<T>; 

  public:

    SlaterDetOperations(int NMO, int NAEA):
        SM_TMats(nullptr) 
    {
      // IWORK: integer buffer for getri/getrf
      IWORK.resize(NMO);

      // local temporary storage
      TMat_NM.resize(extents[NAEA][NMO]);
      TMat_MN.resize(extents[NMO][NAEA]);
      TMat_NN.resize(extents[NAEA][NAEA]);
      TMat_MM.resize(extents[NMO][NMO]);
      TMat_MM2.resize(extents[NMO][NMO]);
      TMat_MM3.resize(extents[NMO][NMO]);

      // reserve enough space in lapack's work array
      // Make sure it is large enough for:
      // 1. getri( TMat_NN )
      WORK.reserve(  ma::getri_optimal_workspace_size(TMat_NN) );
      //  2. geqrf( TMat_NM )
      WORK.reserve(  ma::geqrf_optimal_workspace_size(TMat_NM) );
      //  3. gqr( TMat_NM )
      WORK.reserve(  ma::gqr_optimal_workspace_size(TMat_NM) );
      //  4. gelqf( TMat_MN )
      WORK.reserve(  ma::gelqf_optimal_workspace_size(TMat_MN) );
      //  5. glq( TMat_MN )
      WORK.reserve(  ma::glq_optimal_workspace_size(TMat_MN) );

      // TAU: T used in QR routines 
      TAU.resize(extents[NMO]);

    }

    ~SlaterDetOperations() {}

    SlaterDetOperations(const SlaterDetOperations& other) = delete;
    SlaterDetOperations(SlaterDetOperations&& other) = default;
    SlaterDetOperations& operator=(const SlaterDetOperations& other) = delete;
    SlaterDetOperations& operator=(SlaterDetOperations&& other) = default;

    template<class MatA, class MatB, class MatC>
    T MixedDensityMatrix(const MatA& hermA, const MatB& B, MatC&& C, bool compact=false) {
      int NMO = hermA.shape()[1];
      int NAEA = hermA.shape()[0];
      assert(TMat_NN.num_elements() >= NAEA*NAEA);
      boost::multi_array_ref<T,2> TNN(TMat_NN.data(), extents[NAEA][NAEA]);
      assert(TMat_NM.num_elements() >= NAEA*NMO);
      boost::multi_array_ref<T,2> TNM(TMat_NM.data(), extents[NAEA][NMO]);
      return SlaterDeterminantOperations::base::MixedDensityMatrix<T>(hermA,B,std::forward<MatC>(C),TNN,TNM,IWORK,WORK,compact);
    }

    template<class MatA, class MatC>
    T MixedDensityMatrix(const MatA& A, MatC&& C, bool compact=false) {
      int NMO = A.shape()[0];
      int NAEA = A.shape()[1];
      assert(TMat_NN.num_elements() >= NAEA*NAEA);
      boost::multi_array_ref<T,2> TNN(TMat_NN.data(), extents[NAEA][NAEA]);
      assert(TMat_NM.num_elements() >= NAEA*NMO);
      boost::multi_array_ref<T,2> TNM(TMat_NM.data(), extents[NAEA][NMO]);
      return SlaterDeterminantOperations::base::MixedDensityMatrix_noHerm<T>(A,A,std::forward<MatC>(C),TNN,TNM,IWORK,WORK,compact);
    }

    template<class MatA, class MatB, class MatC>
    T MixedDensityMatrix_noHerm(const MatA& A, const MatB& B, MatC&& C, bool compact=false) {
      int NMO = A.shape()[0];
      int NAEA = A.shape()[1];
      assert(TMat_NN.num_elements() >= NAEA*NAEA);
      boost::multi_array_ref<T,2> TNN(TMat_NN.data(), extents[NAEA][NAEA]);
      assert(TMat_NM.num_elements() >= NAEA*NMO);
      boost::multi_array_ref<T,2> TNM(TMat_NM.data(), extents[NAEA][NMO]);
      return SlaterDeterminantOperations::base::MixedDensityMatrix_noHerm<T>(A,B,std::forward<MatC>(C),TNN,TNM,IWORK,WORK,compact);
    }

    // C must live in shared memory for this routine to work as expected
    template<class MatA, class MatB, class MatC>
    T MixedDensityMatrix(const MatA& hermA, const MatB& B, MatC&& C, communicator& comm, bool compact=false) {
      int NMO = hermA.shape()[1];
      int NAEA = hermA.shape()[0];
      set_shm_buffer(comm,NAEA*(NAEA+NMO));
      assert(SM_TMats->size() >= NAEA*(NAEA+NMO));
      boost::multi_array_ref<T,2> TNN(SM_TMats->data(), extents[NAEA][NAEA]);
      boost::multi_array_ref<T,2> TNM(SM_TMats->data()+NAEA*NAEA, extents[NAEA][NMO]);  
      return SlaterDeterminantOperations::shm::MixedDensityMatrix<T>(hermA,B,std::forward<MatC>(C),TNN,TNM,IWORK,WORK,comm,compact);
    }

    template<class integer, class MatA, class MatB, class MatC, class MatQ>
    T MixedDensityMatrixForWoodbury(const MatA& hermA, const MatB& B, MatC&& C, 
                                    integer* ref, MatQ&& QQ0, bool compact=false) {
      int Nact = hermA.shape()[0];
      int NEL = B.shape()[1];
      int NMO = B.shape()[0];
      assert(hermA.shape()[1]==B.shape()[0]);
      assert(QQ0.shape()[0]==Nact);
      assert(QQ0.shape()[1]==NEL);
      assert(TMat_NN.num_elements() >= NEL*NEL);
      boost::multi_array_ref<T,2> TNN(TMat_NN.data(), extents[NEL][NEL]);
      assert(TMat_NM.num_elements() >= Nact*NEL);
      boost::multi_array_ref<T,2> TAB(TMat_NM.data(), extents[Nact][NEL]);
      assert(TMat_MM.num_elements() >= NMO*NEL);
      boost::multi_array_ref<T,2> TNM(TMat_MM.data(), extents[NEL][NMO]);
      return SlaterDeterminantOperations::base::MixedDensityMatrixForWoodbury<T>(hermA,B,std::forward<MatC>(C),std::forward<MatQ>(QQ0),ref,TNN,TAB,TNM,IWORK,WORK,compact);
    }

    template<class integer, class MatA, class MatB, class MatC, class MatQ>
    T MixedDensityMatrixForWoodbury(const MatA& hermA, const MatB& B, MatC&& C,
                                    integer* ref, MatQ&& QQ0, communicator& comm, bool compact=false) {
      int Nact = hermA.shape()[0];
      int NEL = B.shape()[1];
      int NMO = B.shape()[0];
      assert(hermA.shape()[1]==B.shape()[0]);
      assert(QQ0.shape()[0]==Nact);
      assert(QQ0.shape()[1]==NEL);

      set_shm_buffer(comm,NEL*(NEL+Nact+NMO));
      assert(SM_TMats->size() >= NEL*(NEL+Nact+NMO));
      size_t cnt=0;
      boost::multi_array_ref<T,2> TNN(SM_TMats->data(), extents[NEL][NEL]);
      cnt+=TNN.num_elements();
      boost::multi_array_ref<T,2> TAB(SM_TMats->data()+cnt, extents[Nact][NEL]);
      cnt+=TAB.num_elements();
      boost::multi_array_ref<T,2> TNM(SM_TMats->data()+cnt, extents[NEL][NMO]);
      return SlaterDeterminantOperations::shm::MixedDensityMatrixForWoodbury<T>(hermA,B,std::forward<MatC>(C),std::forward<MatQ>(QQ0),ref,TNN,TAB,TNM,IWORK,WORK,comm,compact);
    }

    template<class integer, class MatA, class MatB, class MatC>
    T MixedDensityMatrixFromConfiguration(const MatA& hermA, const MatB& B, MatC&& C,
                                    integer* ref, bool compact=false) {
      int Nact = hermA.shape()[0];
      int NEL = B.shape()[1];
      int NMO = B.shape()[0];
      assert(hermA.shape()[1]==B.shape()[0]);
      assert(TMat_NN.num_elements() >= NEL*NEL);
      boost::multi_array_ref<T,2> TNN(TMat_NN.data(), extents[NEL][NEL]);
      assert(TMat_NM.num_elements() >= Nact*NEL);
      boost::multi_array_ref<T,2> TAB(TMat_NM.data(), extents[Nact][NEL]);
      assert(TMat_MM.num_elements() >= NMO*NEL);
      boost::multi_array_ref<T,2> TNM(TMat_MM.data(), extents[NEL][NMO]);
      return SlaterDeterminantOperations::base::MixedDensityMatrixFromConfiguration<T>(hermA,B,std::forward<MatC>(C),ref,TNN,TAB,TNM,IWORK,WORK,compact);
    }

    template<class MatA>
    T Overlap(const MatA& A) {
      int NAEA = A.shape()[1];
      assert(TMat_NN.num_elements() >= NAEA*NAEA);
      boost::multi_array_ref<T,2> TNN(TMat_NN.data(), extents[NAEA][NAEA]);
      return SlaterDeterminantOperations::base::Overlap_noHerm<T>(A,A,TNN,IWORK);
    }

    template<class MatA, class MatB>
    T Overlap(const MatA& hermA, const MatB& B) {
      int NAEA = hermA.shape()[0];
      assert(TMat_NN.num_elements() >= NAEA*NAEA);
      boost::multi_array_ref<T,2> TNN(TMat_NN.data(), extents[NAEA][NAEA]);
      return SlaterDeterminantOperations::base::Overlap<T>(hermA,B,TNN,IWORK);
    } 

    template<class MatA, class MatB>
    T Overlap_noHerm(const MatA& A, const MatB& B) {
      int NAEA = A.shape()[1];
      assert(TMat_NN.num_elements() >= NAEA*NAEA);
      boost::multi_array_ref<T,2> TNN(TMat_NN.data(), extents[NAEA][NAEA]);
      return SlaterDeterminantOperations::base::Overlap_noHerm<T>(A,B,TNN,IWORK);
    }

    template<class MatA, class MatB>
    T Overlap(const MatA& hermA, const MatB& B, communicator& comm) { 
      int NAEA = hermA.shape()[0];
      set_shm_buffer(comm,NAEA*NAEA);
      assert(SM_TMats->size() >= NAEA*NAEA);
      boost::multi_array_ref<T,2> TNN(SM_TMats->data(), extents[NAEA][NAEA]);
      return SlaterDeterminantOperations::shm::Overlap<T>(hermA,B,TNN,IWORK,comm);
    }

    // routines for PHMSD
    template<typename integer, class MatA, class MatB, class MatC>
    T OverlapForWoodbury(const MatA& hermA, const MatB& B, integer* ref, MatC&& QQ0) {
      int Nact = hermA.shape()[0];
      int NEL = B.shape()[1];
      assert(hermA.shape()[1]==B.shape()[0]);
      assert(QQ0.shape()[0]==Nact);  
      assert(QQ0.shape()[1]==NEL);  
      assert(TMat_NN.num_elements() >= NEL*NEL);
      assert(TMat_MM.num_elements() >= Nact*NEL);
      boost::multi_array_ref<T,2> TNN(TMat_NN.data(), extents[NEL][NEL]);
      boost::multi_array_ref<T,2> TMN(TMat_MM.data(), extents[Nact][NEL]);
      return SlaterDeterminantOperations::base::OverlapForWoodbury<T>(hermA,B,std::forward<MatC>(QQ0),ref,TNN,TMN,IWORK,WORK);
    }    

    template<typename integer, class MatA, class MatB, class MatC>
    T OverlapForWoodbury(const MatA& hermA, const MatB& B, integer* ref, MatC&& QQ0, communicator& comm) {
      int Nact = hermA.shape()[0];
      int NEL = B.shape()[1];
      assert(hermA.shape()[1]==B.shape()[0]);
      assert(QQ0.shape()[0]==Nact);
      assert(QQ0.shape()[1]==NEL);
      assert(TMat_NN.num_elements() >= NEL*NEL);
      assert(TMat_MM.num_elements() >= Nact*NEL);
      set_shm_buffer(comm,NEL*(Nact+NEL));
      assert(SM_TMats->size() >= NEL*(Nact+NEL));
      boost::multi_array_ref<T,2> TNN(std::addressof(*SM_TMats->data()), extents[NEL][NEL]);
      boost::multi_array_ref<T,2> TMN(std::addressof(*(SM_TMats->data()+NEL*NEL)), extents[Nact][NEL]);
      return SlaterDeterminantOperations::shm::OverlapForWoodbury<T>(hermA,B,std::forward<MatC>(QQ0),ref,TNN,TMN,IWORK,WORK,comm);
    }

    template<class Mat, class MatP1, class MatV>
    void Propagate(Mat&& A, const MatP1& P1, const MatV& V, int order=6) {
      int NMO = A.shape()[0];
      int NAEA = A.shape()[1];
      if(TMat_MN.num_elements() < NMO*NAEA)
        TMat_MN.resize(extents[NMO][NAEA]);
      if(TMat_NM.num_elements() < NMO*NAEA)
        TMat_NM.resize(extents[NAEA][NMO]);
      boost::multi_array_ref<T,2> TMN(TMat_MN.data(), extents[NMO][NAEA]);
      boost::multi_array_ref<T,2> T1(TMat_NM.data(), extents[NMO][NAEA]);
      boost::multi_array_ref<T,2> T2(TMat_MM.data(), extents[NMO][NAEA]);
      ma::product(P1,std::forward<Mat>(A),TMN);
      SlaterDeterminantOperations::base::apply_expM(V,TMN,T1,T2,order);
      ma::product(P1,TMN,std::forward<Mat>(A));
    }

    template<class Mat, class MatP1, class MatV>
    void Propagate(Mat&& A, const MatP1& P1, const MatV& V, communicator& comm, int order=6) {
      int NMO = A.shape()[0];
      int NAEA = A.shape()[1];
      set_shm_buffer(comm,3*NAEA*NMO);
      assert(SM_TMats->size() >= 3*NAEA*NAEA);
      boost::multi_array_ref<T,2> T0(SM_TMats->data(), extents[NMO][NAEA]);
      boost::multi_array_ref<T,2> T1(SM_TMats->data()+NMO*NAEA, extents[NMO][NAEA]);
      boost::multi_array_ref<T,2> T2(SM_TMats->data()+2*NMO*NAEA, extents[NMO][NAEA]);
      if(comm.root()) 
        ma::product(P1,std::forward<Mat>(A),T0);
      comm.barrier();
      SlaterDeterminantOperations::shm::apply_expM(V,T0,T1,T2,comm,order);
      comm.barrier();
      if(comm.root()) 
        ma::product(P1,T0,std::forward<Mat>(A));
      comm.barrier();
    }

    // need to check if this is equivalent to QR!!!
    template<class Mat>
    T Orthogonalize(Mat&& A) {
      T detR = T(1.0);
      ma::gelqf(std::forward<Mat>(A),TAU,WORK);
      for (int i = 0; i < A.shape()[1]; i++) { 
        if (real(A[i][i]) < 0) 
          IWORK[i]=-1; 
        else 
          IWORK[i]=1; 
        detR *= T(IWORK[i])*A[i][i];
      }
      ma::glq(std::forward<Mat>(A),TAU,WORK);
      for(int i=0; i<A.shape()[0]; ++i)
        for(int j=0; j<A.shape()[1]; ++j)
          A[i][j] *= T(IWORK[j]);
      return detR;
    }

  private:

    std::vector<T> WORK;
    std::vector<int> IWORK;

    // Vector used in QR routines 
    TVector TAU;

    // TMat_AB: Local temporary Matrix of dimension [AxB]
    // N: NAEA
    // M: NMO
    TMatrix TMat_NN;
    TMatrix TMat_NM;
    TMatrix TMat_MN;
    TMatrix TMat_MM;
    TMatrix TMat_MM2;
    TMatrix TMat_MM3;

    // shm temporary matrices
    std::unique_ptr<SHM_Buffer> SM_TMats;

    // somewhat sensitive piece of code
    // is this correct?
    void set_shm_buffer(communicator& comm, size_t N) {
      if(SM_TMats == nullptr) {
        SM_TMats = std::move(std::make_unique<SHM_Buffer>(comm,N)); 
      } else if(comm != SM_TMats->getCommunicator()) {
        SM_TMats = std::move(std::make_unique<SHM_Buffer>(comm,N));  
      } else if(SM_TMats->size() < N) {
        SM_TMats->resize(N);
      } 
    }
};

}

}

#endif