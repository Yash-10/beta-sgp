# Approach used for the flux-conservation constraint from http://dx.doi.org/10.1051/0004-6361/201118681 and Bonettini et al. 2009.

import numpy as np

EPSILON = np.finfo(float).eps


def projectDF(b, c, dia, scaling, ccd_sat_level=None, lambda_=0, dlambda_=1, tol_lam=1e-11, biter=0, siter=0, max_projs=1000):
    """
    Equation: min 0.5 * x' * diag(dia) * x - c' * x
                subj to sum(x) = b    
                x >= 0

    """
    c = c.astype(np.float64, copy=False)
    dia = dia.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)

    tol_r = 1e-11 * b

    # Bracketing phase.
    # np.maximum takes element-wise maximum unlike np.max
    x = np.maximum(0, np.divide(c+lambda_, dia))
    if ccd_sat_level is not None:
        x = np.minimum(ccd_sat_level/scaling-EPSILON, x)
    r = np.sum(x) - b

    if abs(r) < tol_r:
        return x

    if r < 0:
        lambdal = lambda_
        rl = r
        lambda_ = lambda_ + dlambda_
        x = np.maximum(0, np.divide(c+lambda_, dia))
        if ccd_sat_level is not None:
            x = np.minimum(ccd_sat_level/scaling-EPSILON, x)
        r = np.sum(x) - b
        while r < 0:
            biter = biter + 1
            lambdal = lambda_
            ## The below commented two lines can also be used as a replacement for the following two lines: from the paper.
            # s = min(1, max(rl/r - 1, 0.1))
            # dlambda_ /= s
            s = np.max([rl/r-1, 0.1])
            dlambda_ = dlambda_ + dlambda_/s

            lambda_ = lambda_ + dlambda_
            rl = r
            x = np.maximum(0, np.divide(c+lambda_, dia))
            if ccd_sat_level is not None:
                x = np.minimum(ccd_sat_level/scaling-EPSILON, x)
            r = np.sum(x) - b
        lambdau = lambda_
        ru = r
    else:
        lambdau = lambda_
        ru = r
        lambda_ = lambda_ - dlambda_
        x = np.maximum(0, np.divide(c+lambda_, dia))
        if ccd_sat_level is not None:
            x = np.minimum(ccd_sat_level/scaling-EPSILON, x)
        r = np.sum(x) - b
        while r > 0:
            biter = biter + 1
            lambdau = lambda_
            s = np.max([ru/r-1, 0.1])
            # Program hangs for some image at the below line. Hence catch warning as an exception and continue.
            try:
                with np.errstate(all='raise'):
                    dlambda_ = dlambda_ + dlambda_/s
            except Exception:
                break
                # raise ValueError("Something bad happened!")
            lambda_ = lambda_ - dlambda_
            ru = r
            x = np.maximum(0, np.divide(c+lambda_, dia))
            if ccd_sat_level is not None:
                x = np.minimum(ccd_sat_level/scaling-EPSILON, x)
            r = np.sum(x) - b
        lambdal = lambda_
        rl = r

    # Check ru and rl
    if abs(ru) < tol_r:
        x = np.maximum(0, np.divide(c+lambdau, dia))
        if ccd_sat_level is not None:
            x = np.minimum(ccd_sat_level/scaling-EPSILON, x)
        return x
    if abs(rl) < tol_r:
        x = np.maximum(0, np.divide(c+lambdal, dia))
        if ccd_sat_level is not None:
            x = np.minimum(ccd_sat_level/scaling-EPSILON, x)
        return x

    # Secant phase
    s = 1 - rl/ru
    dlambda_ = dlambda_/s
    lambda_ = lambdau - dlambda_
    x = np.maximum(0, np.divide(c+lambda_, dia))
    if ccd_sat_level is not None:
        x = np.minimum(ccd_sat_level/scaling-EPSILON, x)
    r = np.sum(x) - b
    maxit_s = max_projs - biter

    # Main loop
    while abs(r) > tol_r and dlambda_ > tol_lam * (1 + abs(lambda_)) and siter < maxit_s:
        siter = siter + 1
        if r > 0:
            if s <= 2:
                lambdau = lambda_
                ru = r
                s = 1-rl/ru
                dlambda_ = (lambdau-lambdal)/s
                lambda_ = lambdau - dlambda_
            else:
                s = np.max([ru/r-1, 0.1])
                dlambda_ = (lambdau-lambda_) / s
                lambda_new = np.max([lambda_-dlambda_, 0.75*lambdal+0.25*lambda_])
                lambdau = lambda_
                ru = r
                lambda_ = lambda_new
                x = (lambdau-lambdal) / (lambdau-lambda_)
        else:
            if s >= 2:
                lambdal = lambda_
                rl = r
                s = 1-rl/ru
                dlambda_ = (lambdau-lambdal)/s
                lambda_ = lambdau - dlambda_
            else:
                s = np.max([rl/r-1, 0.1])
                dlambda_ = (lambda_-lambdal) / s
                lambda_new = np.min([lambda_+dlambda_, 0.75*lambdau+0.25*lambda_])
                lambdal = lambda_
                rl = r
                lambda_ = lambda_new
                s = (lambdau-lambdal) / (lambdau-lambda_)

        x = np.maximum(0, np.divide(c+lambda_, dia))
        if ccd_sat_level is not None:
            x = np.minimum(ccd_sat_level/scaling-EPSILON, x)
        r = np.sum(x) - b

    return x
