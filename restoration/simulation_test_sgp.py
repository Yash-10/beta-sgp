import numpy as np
from scipy.io import loadmat
from sgp import sgp, sgp_betaDiv

def _plot_helper(image, obj, deconv):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image, origin='lower')
    ax[0].set_title('Degraded image')
    ax[1].imshow(obj, origin='lower')
    ax[1].set_title('Ground truth')
    ax[2].imshow(deconv, origin='lower')
    ax[2].set_title('Deconvolved')
    plt.savefig('test_sgp.png')
    plt.show()

def test_sgp_with_sgpdec_ngc(plot=False):
    ngc7027 = loadmat('simulated_test/data/NGC7027_255.mat')

    image = ngc7027['gn']
    psf = ngc7027['psf']
    bkg = ngc7027['bg'][0][0]
    obj = ngc7027['obj']

    deconv, _, _, _, _ = sgp(image, psf, bkg, init_recon=3, stop_criterion=1, MAXIT=27)

    e = deconv - obj
    obj_sum = np.sum(obj * obj)
    rel_err = np.sqrt(np.sum(e * e) / obj_sum)

    if plot:
        _plot_helper(image, obj, deconv)

    return deconv, rel_err, image, obj


def test_sgp_with_sgpdec_satellite(plot=False):
    sat = loadmat('simulated_test/data/satellite_25500.mat')

    image = sat['gn']
    psf = sat['psf']
    bkg = sat['bg'][0][0]
    obj = sat['obj']

    deconv, _, _, _, _ = sgp(image, psf, bkg, init_recon=3, stop_criterion=1, MAXIT=332)

    e = deconv - obj
    obj_sum = np.sum(obj * obj)
    rel_err = np.sqrt(np.sum(e * e) / obj_sum)

    if plot:
        _plot_helper(image, obj, deconv)

    return deconv, rel_err, image, obj


def test_sgp_betaDiv_with_sgpdec_ngc(plot=False, do_sampling=False):
    ngc7027 = loadmat('simulated_test/data/NGC7027_255.mat')

    image = ngc7027['gn']
    psf = ngc7027['psf']
    bkg = ngc7027['bg'][0][0]
    obj = ngc7027['obj']

    if do_sampling:
        best_beta_init, best_rel_err = None, np.Inf

        np.random.seed(42)
        rands = []
        for _ in range(30):
            rands.append(
                np.random.normal(loc=1, scale=0.05)
            )

        for rand in rands:
            betaParam = rand
            deconv, _, _, _, _ = sgp_betaDiv(  # TODO: we need to set maxit based on which iter gives min error...27 was for normal SGP.
                image, psf, bkg, init_recon=3, stop_criterion=1, MAXIT=27,
                betaParam=betaParam, lr=1e-3, lr_exp_param=0.1, schedule_lr=True,
                adapt_beta=True
            )

            e = deconv - obj
            obj_sum = np.sum(obj * obj)
            rel_err = np.sqrt(np.sum(e * e) / obj_sum)

            if plot:
                _plot_helper(image, obj, deconv)

            print(f'Beta-init: {betaParam}, rel_err: {rel_err}')

            if rel_err < best_rel_err:
                best_rel_err = rel_err
                best_beta_init = betaParam

        print(f'Best beta_init parameter [rel_err = {best_rel_err}] = {best_beta_init}. Running with this beta_init...')
    else:
        best_beta_init = 0.9887296104546054

    deconv, _, _, _, _ = sgp_betaDiv(  # TODO: we need to set maxit based on which iter gives min error...27 was for normal SGP.
            image, psf, bkg, init_recon=3, stop_criterion=1, MAXIT=27,
            betaParam=best_beta_init, lr=1e-3, lr_exp_param=0.1, schedule_lr=True,
            adapt_beta=False
        )

    e = deconv - obj
    obj_sum = np.sum(obj * obj)
    rel_err = np.sqrt(np.sum(e * e) / obj_sum)

    return deconv, rel_err, image, obj

def test_sgp_betaDiv_with_sgpdec_satellite(plot=False, do_sampling=False):
    sat = loadmat('simulated_test/data/satellite_25500.mat')

    image = sat['gn']
    psf = sat['psf']
    bkg = sat['bg'][0][0]
    obj = sat['obj']

    if do_sampling:
        best_beta_init, best_rel_err = None, np.Inf

        np.random.seed(42)
        rands = []
        for _ in range(30):
            rands.append(
                np.random.normal(loc=1, scale=0.01)
            )

        for rand in rands:
            betaParam = rand
            deconv, _, _, _, _ = sgp_betaDiv(
                image, psf, bkg, init_recon=3, stop_criterion=1, MAXIT=332,
                betaParam=betaParam, lr=1e-3, lr_exp_param=0.1, schedule_lr=True,
                adapt_beta=True
            )

            e = deconv - obj
            obj_sum = np.sum(obj * obj)
            rel_err = np.sqrt(np.sum(e * e) / obj_sum)

            if plot:
                _plot_helper(image, obj, deconv)

            print(f'Beta-init: {betaParam}, rel_err: {rel_err}')

            if rel_err < best_rel_err:
                best_rel_err = rel_err
                best_beta_init = betaParam

        print(f'Best beta_init parameter [rel_err = {best_rel_err}] = {best_beta_init}. Running with this beta_init...')
    else:
        # best_beta_init = 1.0002419622715661
        best_beta_init = 1.0001

    deconv, _, _, _, _ = sgp_betaDiv(
        image, psf, bkg, init_recon=3, stop_criterion=1, MAXIT=332,
        betaParam=best_beta_init, lr=1e-3, lr_exp_param=0.1, schedule_lr=True,
        adapt_beta=False
    )

    e = deconv - obj
    obj_sum = np.sum(obj * obj)
    rel_err = np.sqrt(np.sum(e * e) / obj_sum)

    if plot:
        _plot_helper(image, obj, deconv)

    return deconv, rel_err, image, obj

if __name__ == "__main__":
    ngc1, relngc1, orig_ngc1, objngc1 = test_sgp_with_sgpdec_ngc(plot=False)
    sat1, relsat1, orig_sat1, objsat1 = test_sgp_with_sgpdec_satellite(plot=False)
    ngc2, relngc2, orig_ngc2, objngc2 = test_sgp_betaDiv_with_sgpdec_ngc(plot=False, do_sampling=False)
    sat2, relsat2, orig_sat2, objsat2 = test_sgp_betaDiv_with_sgpdec_satellite(plot=False, do_sampling=False)

    # Plotting results.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(orig_ngc1)
    ax[0].set_title('Original (degraded) image', fontsize=10)
    ax[1].imshow(ngc1)
    ax[1].set_title(f'Restored (SGP), Error = {relngc1:.4f}', fontsize=10)
    ax[2].imshow(ngc2)
    ax[2].set_title(f'Restored (beta-SGP), Error = {relngc2:.4f}', fontsize=10)
    ax[3].imshow(objngc1)
    ax[3].set_title('Ground-truth', fontsize=10)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[3].set_xticks([])
    ax[3].set_yticks([])

    plt.show()

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(orig_sat1)
    ax[0].set_title('Original (degraded) image', fontsize=10)
    ax[1].imshow(sat1)
    ax[1].set_title(f'Restored (SGP), Error = {relsat1:.4f}', fontsize=10)
    ax[2].imshow(sat2)
    ax[2].set_title(f'Restored (beta-SGP), Error = {relsat2:.4f}', fontsize=10)
    ax[3].imshow(objsat1)
    ax[3].set_title('Ground-truth', fontsize=10)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[3].set_xticks([])
    ax[3].set_yticks([])

    plt.show()
