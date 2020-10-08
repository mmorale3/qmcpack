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
// M[i][p][q] = T[iexcit[i][p+nex]][iexcit[i][q]]
__global__ void kernel_extract_overlap_matrix(int ndet,
                                              int nex,
                                              int nmo,
                                              int const* iexcit,
                                              thrust::complex<double> const* T,
                                              thrust::complex<double>* M)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int z = blockIdx.z*blockDim.z + threadIdx.z;
  if (x >= ndet || y >= nex || z >= nex)
    return;
  int ip = iexcit[x*2*nex + y + nex];
  int iq = iexcit[x*2*nex + z];
  /*printf("%d %d %d %d %d %d\n", x, y, z, x*nex*nex + y*nex + z, ip, iq);*/
  /*printf("%d %f\n", x*nex*nex + y*nex + z, T[ip*nmo+iq].real());*/
  M[x*nex*nex + y*nex + z] = T[ip*nmo + iq];
}

void extract_overlap_matrix(int ndet,
                            int nex,
                            int nmo,
                            int const* iexcit,
                            std::complex<double> const* T,
                            std::complex<double> *M)
{
  int xblock_dim = 8;
  int grid_dim_y = (nex + xblock_dim - 1) / xblock_dim;
  int grid_dim_x = (ndet + xblock_dim - 1) / xblock_dim;
  dim3 grid_dim(grid_dim_x, grid_dim_y, nex);
  dim3 block_dim(xblock_dim,xblock_dim,xblock_dim);
  kernel_extract_overlap_matrix<<<grid_dim, block_dim>>>(ndet, nex, nmo, iexcit,
                                                 reinterpret_cast<thrust::complex<double> const*>(T),
                                                 reinterpret_cast<thrust::complex<double>*>(M));
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

}
