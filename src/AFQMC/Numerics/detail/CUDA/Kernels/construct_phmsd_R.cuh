#ifndef AFQMC_CONSTRUCT_PHMSD_R_H
#define AFQMC_CONSTRUCT_PHMSD_R_H
#include <complex>

namespace kernels
{
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
                             std::complex<double>* R);

}
#endif
