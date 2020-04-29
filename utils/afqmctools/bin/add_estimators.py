#! /usr/bin/env python3

import argparse
from mpi4py import MPI
from os import getcwd 
import scipy.sparse
import sys
from time import asctime
try:
    from pyscf.pbc.lib.chkfile import load_cell
except ImportError:
    print("Cannot find pyscf.")
    sys.exit()
from afqmctools.utils.misc import get_git_hash
from afqmctools.utils.pyscf_utils import (
        load_from_pyscf_chk,
        load_from_pyscf_chk_mol
        )
from afqmctools.hamiltonian.supercell import write_rhoG_supercell
from afqmctools.hamiltonian.kpoint import write_rhoG_kpoints

def parse_args(args, comm):
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

    if comm.rank == 0:
        parser = argparse.ArgumentParser(description = __doc__)
        parser.add_argument('-i', '--input', dest='chk_file', type=str,
                            default=None, help='Input pyscf .chk file.')
        parser.add_argument('-o', '--output', dest='hdf_file',
                            type=str, default='estim.h5',
                            help='Output h5 file name for QMCPACK estimator file.')
        parser.add_argument('-q', '--qmcpack-input', dest='qmc_input',
                            type=str, default=None,
                            help='Generate skeleton QMCPACK input file.')
        parser.add_argument('-t', '--type', dest='type', type=str,
                            default=None, help='List of estimators.')
        parser.add_argument('-c', '--cut', dest='Gcut', type=float,
                            default=5.0, help='Kinetic energy cutoff for rho_G.')
        parser.add_argument('-k', '--kpoint', dest='kpoint_sym',
                            action='store_true', default=False,
                            help='Generate explicit kpoint dependent integrals.')
        parser.add_argument('-a', '--ao', dest='ortho_ao',
                            action='store_true', default=False,
                            help='Transform to ortho AO basis. Default assumes '
                            'we work in MO basis')
        parser.add_argument('-r', '--real', dest='write_real',
                            action='store_true', default=False,
                            help='Write data as real numbers')
        parser.add_argument('-v', '--verbose', action='count', default=0,
                            help='Verbose output.')

        options = parser.parse_args(args)
    else:
        options = None
    options = comm.bcast(options, root=0)

    if (not options.chk_file) or (not options.type):
        if comm.rank == 0:
            parser.print_help()
        sys.exit()

    return options

def main(args):
    """Generate hdf5/xml input for QMCPACK estimators from pyscf checkpoint file.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """
    comm = MPI.COMM_WORLD
    options = parse_args(args, comm)
    if comm.rank == 0:
        cwd = getcwd()
        sha1 = get_git_hash()
        date_time = asctime()
        print(" # Generating QMCPACK input from PYSCF checkpoint file.")
        print(" # git sha1: {}".format(sha1))
        print(" # Date/Time: {}".format(date_time))
        print(" # Working directory: {}".format(cwd))

    try:
        obj = load_cell(options.chk_file)
        pbc = True
    except NameError:
        pbc = False
        print(" load failed ")

    if pbc:
        scf_data = load_from_pyscf_chk(options.chk_file, orthoAO=options.ortho_ao)
    else:
        if comm.size > 1:
            if comm.rank == 0:
                print(" # Error molecular integral generation must be done "
                      "in serial.")
            sys.exit()
        scf_data = load_from_pyscf_chk_mol(options.chk_file)

    # turn this into a list that can be parsed, to avoid having to call many times
    if options.type == 'rho_G':
        if pbc:
            if comm.rank == 0 and options.verbose:
                print(" # Generating rho(G) object.")
            if options.kpoint_sym:
                write_rhoG_kpoints(comm, scf_data, options.hdf_file, options.Gcut,
                                verbose=options.verbose,
                                ortho_ao=options.ortho_ao) 
            else:
                write_rhoG_supercell(comm, scf_data, options.hdf_file, options.Gcut,
                                verbose=options.verbose,
                                ortho_ao=options.ortho_ao, 
                                write_real=options.write_real)
        else:
            print(" Error: rho_G requires pbc calculation.")
    elif options.type == 'elecloc':
        print(" Not implemented yet. ")
    elif options.type == 'orbR':
        print(" Not implemented yet. ")
    elif options.type == 'orbG':
        print(" Not implemented yet. ")
    elif options.type == 'atomic_overlaps':
        print(" Not implemented yet. ")
    else:
        print("Unknown estimator type:"+options.type)

    if comm.rank == 0:
        print("\n # Finished.")

if __name__ == '__main__':

    main(sys.argv[1:])
