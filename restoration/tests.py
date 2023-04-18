from sgp import sgp, sgp_betaDiv, betaDiv, betaDivDeriv, betaDivDerivwrtY

import numpy as np
import torch
from scipy.io import loadmat
from functools import partial
from torchnmf.metrics import beta_div

def test_betadivergence_value_matches(beta=1.5):
    """This uses the torchnmf library which needs to be installed. See https://pytorch-nmf.readthedocs.io/"""
    torch.manual_seed(101)
    x1 = torch.rand(20)
    torch.manual_seed(1001)
    y1 = torch.rand(20)

    pytorch_nmf_betadiv = beta_div(x1, y1, beta=beta)
    our_betadiv = betaDiv(x1.numpy(), y1.numpy(), betaParam=beta)

    assert np.isclose(pytorch_nmf_betadiv, our_betadiv)

def test_betaDivDerivwrtY_matches():
    # Setup
    data = loadmat('simulated_test/data/NGC7027_255.mat')
    psf = data['psf'].flatten()
    bkg = data['bg'].flatten()
    gn = data['gn'].flatten()
    x = data['obj'].flatten()

    ONE = np.ones(gn.size)

    TF = np.fft.fftn(np.fft.fftshift(psf))
    CTF = np.conj(TF)
    def afunction(x, TF, dimensions):
        x = np.reshape(x, dimensions)
        out = np.real(np.fft.ifftn(
            np.multiply(TF, np.fft.fftn(x))
        ))
        out = out.flatten()
        return out

    A = partial(afunction, TF=TF, dimensions=psf.shape)
    AT = partial(afunction, TF=CTF, dimensions=psf.shape)

    x_tf = A(x=x)
    den = x_tf + bkg
    temp = np.divide(gn, den)
    kldiv_gradient = ONE - AT(x=temp)

    # Calculate gradient using beta divergence function
    betadiv_gradient = betaDivDerivwrtY(AT, den, gn, betaParam=1)

    assert np.allclose(kldiv_gradient, betadiv_gradient)

def test_betaDivDeriv_matches(beta=1.7):
    # Some random tensors.
    den = torch.tensor([1, 2.053, 4.5, 7.9, 1.5], requires_grad=True)
    gn = torch.tensor([9.3, 2.5, 4.5, 7.9, 1.5], requires_grad=True)
    betaParam = torch.tensor(beta, requires_grad=True)

    betadiv_bgrad = betaDivDeriv(den.detach().numpy(), gn.detach().numpy(), betaParam=betaParam.detach().numpy()).sum()

    # Using PyTorch.
    f = beta_div(den, gn, beta=betaParam)
    print(f)
    f.backward()
    betadiv_gradient = betaParam.grad.item()

    assert np.isclose(betadiv_bgrad, betadiv_gradient)


if __name__ == "__main__":
    test_betadivergence_value_matches()
    test_betaDivDerivwrtY_matches()
    test_betaDivDeriv_matches()
