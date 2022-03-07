def mos_sys18(uttid):
    sysid = uttid.split('_')[0]
    return sysid

def mos_sys20(uttid):
    sysid = uttid.split('-')[0]
    return sysid

def mos_bvcc(uttid):
    sysid = '-'.join(uttid.split('-')[:2])
    return sysid

def de_dup_solve(uttid):
    return '_'.join(uttid.split('_')[:-1])

def all(uttid):
    return 'all'

def utt(uttid):
    return uttid.split('#')[0]

def svs18(uttid):
    pairid = uttid.split('#')[0]
    sys1 = mos_sys18(pairid.split('&')[0])
    sys2 = mos_sys18(pairid.split('&')[1])
    return sys1 + sys2

