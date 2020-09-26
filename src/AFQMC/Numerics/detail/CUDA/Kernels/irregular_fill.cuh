#ifndef AFQMC_IRREGULAR_FILL_HPP
#define AFQMC_IRREGULAR_FILL_HPP

#include <complex>

namespace kernels
{

void irregular_fill(int ndet,
                    int nex,
                    int nmo,
                    int const* iexcit,
                    std::complex<double> const* T,
                    std::complex<double> *M);

}
#endif
