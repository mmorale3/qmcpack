#include <complex>
#include <cuda.h>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "AFQMC/Numerics/detail/CUDA/Kernels/cuda_settings.h"
#define ENABLE_CUDA 1
#include "AFQMC/Memory/CUDA/cuda_utilities.h"

namespace kernels
{
// extract overlap matrix (M) for PHMSD
__global__ void kernel_construct_phmsd_R(int ndet,
                                         int nex,
                                         int nact,
                                         int nelec,
                                         int nmo,
                                         int const* iexcit,
                                         int const* orbs,
                                         thrust::complex<double> const* weights,
                                         thrust::complex<double> const* T,
                                         thrust::complex<double> const* I,
                                         thrust::complex<double>* Rbuff,
                                         thrust::complex<double>* R)
{
  int idet = blockIdx.z;
  int p = threadIdx.x + blockDim.x * blockIdx.x;
  int q = threadIdx.y + blockDim.y * blockIdx.y;
  if (idet >= ndet || p >= nelec || q >= nelec)
    return;
  // for (i = 0 : nelec) R[nd][i][i] = w
  if (p < nelec && p == q) {
    Rbuff[idet*nact*nelec + p*nelec + q] = weights[idet];
  }
  __syncthreads();
  // for (i = 0 : nelec) R[nd][p][i] = w
 // NEED an extra index !~!!!
  if (p < nex && q < nex) {
    for (int i = 0; i < nelec; i++) {
      double* re = reinterpret_cast<double*>(&Rbuff[idet*nact*nelec+p*nelec+i]);
      /*double* re = &(Rbuff[idet*nact*nelec + p*nelec + i].real());*/
      /*double* im = &(Rbuff[idet*nact*nelec + p*nelec + i].imag());*/
      thrust::complex<double> val = -(weights[idet] * I[idet*nex*nex + p*nex + q] * T[q*nmo + i]);
      double v_re = val.real();
      double v_im = val.imag();
          /*, Rbuff[idet*nact*nelec+p*nelec+i].imag());*/
      /*if (idet == 0) printf("%d %d %d %f %f %f %f\n", i, p, q, *re, *(re+1), v_re, v_im);*/
      atomicAdd(re, v_re);
      atomicAdd(re + 1, v_im);
      /*if (idet == 0) printf("%d %d %d %f %f %f %f\n", i, p, q, *re, *(re+1), v_re, v_im);*/
      /*atomicAdd(im, v_im);*/
    }
  }
  __syncthreads();
  if (p < nex && q < nex) {
    Rbuff[idet*nact*nelec + p*nelec + q] += weights[idet] * I[idet*nex*nex + p*nex + q];
  }
  __syncthreads();
  // Reduce into R[p,q] = \sum_idet R[idet, p, q]
  /*double* re = reinterpret_cast<double*>(&R[p*nelec+q]);*/
  /*if (p < nact && q < nelec) {*/
    /*thrust::complex<double> val = Rbuff[idet*nact*nelec + p*nelec + q];*/
    /*atomicAdd(re, val.real());*/
    /*atomicAdd(re + 1, val.imag());*/
  /*}*/
}

void construct_phmsd_R(int ndet,
                             int nex,
                             int nact,
                             int nelec,
                             int nmo,
                             int const* iexcit,
                             int const* orbs,
                             std::complex<double> const* weights,
                             std::complex<double> const* T,
                             std::complex<double> const* I,
                             std::complex<double>* Rbuff,
                             std::complex<double>* R)
{
  int xblock_dim = 16;
  int grid_dim_x = (nelec + xblock_dim - 1) / xblock_dim;
  int grid_dim_y = (nelec + xblock_dim - 1) / xblock_dim;
  dim3 grid_dim(grid_dim_x, grid_dim_y, ndet);
  dim3 block_dim(xblock_dim, xblock_dim);
  kernel_construct_phmsd_R<<<grid_dim, block_dim>>>(ndet, nex, nact, nelec, nmo, iexcit, orbs,
                                                    reinterpret_cast<thrust::complex<double> const*>(weights),
                                                    reinterpret_cast<thrust::complex<double> const*>(T),
                                                    reinterpret_cast<thrust::complex<double> const*>(I),
                                                    reinterpret_cast<thrust::complex<double>*>(Rbuff),
                                                    reinterpret_cast<thrust::complex<double>*>(R));
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

}
