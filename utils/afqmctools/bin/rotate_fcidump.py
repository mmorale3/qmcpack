#!/usr/bin/env python

import argparse
import scipy.sparse
import scipy.linalg
import sys
import time
import numpy
from afqmctools.hamiltonian.mol import (
        write_qmcpack_cholesky
        )
from afqmctools.hamiltonian.converter import read_fcidump
from afqmctools.hamiltonian.mol import ao2mo_chol 
from afqmctools.utils.linalg import modified_cholesky_direct
from afqmctools.hamiltonian.converter import (
        write_fcidump,
        )


def parse_args(args):
    """Parse command-line arguments.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    options : :class:`argparse.ArgumentParser`
        Command line arguments.
    """

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('-i', '--input', dest='input_file', type=str,
                        default=None, help='Input FCIDUMP file.')
    parser.add_argument('-r', '--rdm', dest='rdm_file', type=str,
                        default=None, help='ASCII file with rdm.')
    parser.add_argument('-n', '--rot', dest='rot_file', type=str,
                        default='rot.npz', help='File with rotation matrix (numpy.save).')
    parser.add_argument('-o', '--output', dest='output_file',
                        type=str, default='fcidump.h5',
                        help='Output FCIDUMP file..')
    parser.add_argument('--write-complex', dest='write_complex',
                        action='store_true', default=False,
                        help='Output integrals in complex format.')
    parser.add_argument('-t', '--cholesky-threshold', dest='thresh',
                        type=float, default=1e-6,
                        help='Cholesky convergence threshold.')
    parser.add_argument('-c', '--cutoff', dest='tol',
                        type=float, default=1e-8,
                        help='Cutoff for integrals.')
    parser.add_argument('-s', '--symmetry', dest='symm',
                        type=int, default=8,
                        help='Symmetry of integral file (1,4,8).')
    parser.add_argument('-m', '--cmax', dest='cmax',
                        type=int, default=30,
                        help='Maximum number of cholesky vectors, max = cmax*M.')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='Verbose output.')

    options = parser.parse_args(args)

    if not options.input_file or not options.rdm_file:
        parser.print_help()
        sys.exit(1)

    return options

def get_items(line,a):
    for k in line:
        if k != '':
            a.append(k)

def read_rdm(filename, N):

    R = numpy.zeros((N,N),dtype=float)

    with open(filename) as f:
        line = f.readline()
        a=[]
        get_items(line.strip().split(' '),a)
        assert(len(a) == 1)
        assert(int(a[0]) == N)
        line = f.readline()
        while line:
            a=[]
            get_items(line.strip().split(' '),a)
            assert(len(a) == 3)
            R[ int(a[0]), int(a[1]) ] = float(a[2])
            line = f.readline()
    return R

def main(args):
    """Convert FCIDUMP to QMCPACK readable Hamiltonian format.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """
    options = parse_args(args)
    (hcore, eri, ecore, nelec) = read_fcidump(options.input_file,
                                              symmetry=options.symm,
                                              verbose=options.verbose)
    norb = hcore.shape[-1]

    # If the ERIs are complex then we need to form M_{(ik),(lj}} which is
    # Hermitian. For real integrals we will have 8-fold symmetry so trasposing
    # will have no effect.
    eri = numpy.transpose(eri,(0,1,3,2))

    chol = modified_cholesky_direct(eri.reshape(norb**2,norb**2),
                                    options.thresh, options.verbose,
                                    cmax=options.cmax).T.copy()

    rdm = read_rdm(options.rdm_file,hcore.shape[0])

    w,vr = scipy.linalg.eig(rdm)

    numpy.savez(options.rot_file,w,vr)

    hcore_ = numpy.dot(vr.T, numpy.dot(hcore, vr))

    chol = chol.T.copy()
    ao2mo_chol(chol,vr)
    chol = chol.T.copy()

    # need to output vr to be able to rotate back to original basis

    write_fcidump(options.output_file,
                  hcore_,
                  scipy.sparse.csr_matrix(chol),
                  ecore,
                  hcore.shape[0],
                  nelec,
                  tol=options.tol,
                  sym=options.symm,
                  cplx=False,
                  paren=False)

if __name__ == '__main__':
    main(sys.argv[1:])
