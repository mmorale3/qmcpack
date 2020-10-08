#ifndef AFQMC_EXTRACT_OVERLAP_MATRIX
#define AFQMC_EXTRACT_OVERLAP_MATRIX

#include <complex>

namespace kernels
{

void extract_overlap_matrix(int ndet,
                    int nex,
                    int nmo,
                    int const* iexcit,
                    std::complex<double> const* T,
                    std::complex<double> *M);

}
#endif
