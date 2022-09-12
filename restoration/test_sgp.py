import numpy as np
from scipy.io import loadmat
from sgp import sgp

def test_sgp_with_sgpdec_ngc():
    ngc7027 = loadmat('data/NGC7027_255.mat')

    image = ngc7027['gn']
    psf = ngc7027['psf']
    bkg = ngc7027['bg'][0][0]
    obj = ngc7027['obj']

    deconv, iter_, _, _ = sgp(image, psf, bkg, init_recon=3, stop_criterion=1, MAXIT=27)

    e = deconv - obj
    obj_sum = np.sum(obj * obj)
    rel_err = np.sqrt(np.sum(e * e) / obj_sum)

    print(rel_err)

def test_sgp_with_sgpdec_satellite():
    sat = loadmat('data/satellite_25500.mat')

    image = sat['gn']
    psf = sat['psf']
    bkg = sat['bg'][0][0]
    obj = sat['obj']

    deconv, iter_, _, _ = sgp(image, psf, bkg, init_recon=3, stop_criterion=1, MAXIT=332)

    e = deconv - obj
    obj_sum = np.sum(obj * obj)
    rel_err = np.sqrt(np.sum(e * e) / obj_sum)

    print(rel_err)

if __name__ == "__main__":
    test_sgp_with_sgpdec_ngc()
    test_sgp_with_sgpdec_satellite()
