import struct
import numpy

def read_spinRDM(fname,norb):
    # The 2RDMs written by "SHCIrdm::saveRDM" in DICE
    # are written as E2[i1,j2,k2,l2]
    # and are stored here as E2[i1,j2,l2,k2]
    # This is done with SQA in mind.
    fil=open(fname,"rb")
    print("     [fil.seek: How dangerous is that??]")
    fil.seek(53)  # HOW DANGEROUS IS THAT ???!?!?!?
    spina=-1
    spinb=-1
    spinc=-1
    spind=-1
    ab='ab'
    rdmaa=numpy.zeros((norb,norb,norb,norb), order='C')
    rdmbb=numpy.zeros((norb,norb,norb,norb), order='C')
    rdmab=numpy.zeros((norb,norb,norb,norb), order='C')
    for a in range(2*norb):
        spina=(spina+1)%2
        for b in range(a+1):
            spinb=(spinb+1)%2
            for c in range(2*norb):
                spinc=(spinc+1)%2
                for d in range(c+1):
                    spind=(spind+1)%2
                    A,B,C,D=int(a/2.),int(b/2.),int(c/2.),int(d/2.)
                    (value,)=struct.unpack('d',fil.read(8))
                    if spina==spinb and spina==spinc and spina==spind:
                        if spina==0:
                            rdmaa[A,C, B, D]+=value
                            rdmaa[B,D, A, C]+=value
                            rdmaa[A,D, B, C]-=value
                            rdmaa[B,C, A, D]-=value
                        else:
                            rdmbb[A,C, B, D]+=value
                            rdmbb[B,D, A, C]+=value
                            rdmbb[A,D, B, C]-=value
                            rdmbb[B,C, A, D]-=value
                    elif (spina==spinb and spinc==spind)\
                       or(spina==spinc and spinb==spind)\
                       or(spina==spind and spinb==spinc):
                        if (spina==spind and spinb==spinc):
                            if spina==0:
                                rdmab[A,D ,B, C]-=value
                            else:
                                rdmab[B,C, A, D]-=value
                        if (spina==spinc and spinb==spind):
                            if spina==0:
                                rdmab[A,C ,B, D]+=value
                            else:
                                rdmab[B,D ,A, C]+=value
                spind=-1
            spinc=-1
        spinb=-1
    spina=-1
    try:
        (value,)=struct.unpack('c',fil.read(1))
        print("     [MORE bytes TO READ!]")
    except:
        print("     [at least, no more bytes to read!]")
      #exit(0)
    fil.close()
    return rdmaa,rdmab,rdmbb
