"""Profiled multinomial Fisher geometry and truth-side attainable-gain bounds."""
from __future__ import annotations

import numpy as np

from scripts.high_n_amplitude import amp_data
from scripts.high_n_amplitude.amp_control import centered_action, conditional_fidelity


def cell_jacobian(p, eta, meta):
    """Four-cell probability derivatives in p, log efficiency, and log idler background."""
    keys = ("chi", "eta_signal", "background", "background_idler", "n_max")
    c = amp_data._cell_polynomial(*(meta[k] for k in keys))
    pi = amp_data.population_probabilities(p, eta, meta)
    derivative = np.polynomial.polynomial.polyder(c.T, axis=0)
    dp = eta[..., None] * np.moveaxis(np.polynomial.polynomial.polyval(eta*p, derivative), 0, -1)
    # Idler background flips either of the two no-idler cells independently.
    db = np.stack((-pi[..., 0], -pi[..., 1], pi[..., 0], pi[..., 1]), axis=-1)
    db *= meta["background_idler"]/(1-meta["background_idler"])
    return pi, dp, np.stack((p[..., None]*dp, db), axis=-1)


def gain_derivative(p, u):
    def derivative(v):
        e = np.exp(v)
        total = (np.sqrt(p)*e).sum(-1, keepdims=True)
        denom = (p*e**2).sum(-1, keepdims=True)
        f = total**2/(p.shape[-1]*denom)
        return total*e/(p.shape[-1]*denom*np.sqrt(p)) - f*e**2/denom
    return derivative(u)-derivative(np.zeros_like(u))


def score_uncertainty(p, eta, meta, u):
    """Variance of proposed gain, evaluated with action fixed at its observed value.

    Inverting the constrained arrowhead information profiles shared nuisance
    coordinates with no prior. Projection onto sum(dp)=0 removes the simplex gauge.
    The resulting Wald penalty is a local ranking statistic; CAL evaluates actual risk.
    """
    pi, dp, dn = cell_jacobian(p, eta, meta)
    precision = meta["trials"]/pi
    inv_d = 1./(precision*dp**2).sum(-1)
    cross = ((precision*dp)[..., None]*dn).sum(-2)
    c = np.einsum('bik,bikl,bikm->blm', precision, dn, dn, optimize=True)
    def project(v):
        return v-(v*inv_d[..., None]).sum(1,keepdims=True)/inv_d.sum(1)[:,None,None]
    cross_c = project(cross)
    grad = project(gain_derivative(p,u)[..., None])[...,0]
    known = (inv_d*grad**2).sum(-1)
    schur = c - np.einsum('bi,bil,bim->blm',inv_d,cross_c,cross_c)
    eig, vec = np.linalg.eigh(schur)
    threshold = 1e-10*np.maximum(np.max(eig,axis=-1,keepdims=True),1e-30)
    if np.any(eig < -threshold):
        raise ArithmeticError("Negative nuisance information eigenvalue")
    t = np.einsum('bi,bi,bil,blm->bm',inv_d,grad,cross_c,vec)
    support = eig > threshold
    extra = np.divide(t*t,eig,out=np.zeros_like(eig),where=support).sum(-1)
    unresolved = ((~support)&(abs(t)>1e-8)).any(-1)
    return np.sqrt(known), np.where(unresolved,np.inf,np.sqrt(known+extra)), support.sum(-1)


def reachable_bounds(p, bound=0.25):
    """Box-relaxed Rayleigh maximum (upper) and centered feasible score (lower).

    For positive y in [sqrt(p)e^-b,sqrt(p)e^b], F = sum(y)^2/(N sum(y^2)).
    Its best direction is the point of the scaled box nearest the all-ones
    unit vector. Componentwise projection clips a common level k. Optimal
    scale obeys sum y_i(y_i-k)=0; the monotone piecewise linear equation is
    bisected. Removing the zero-mean restriction gives an upper bound.
    """
    a = np.sqrt(p)
    lo, hi = a*np.exp(-bound), a*np.exp(bound)
    left, right = lo.min(-1, keepdims=True), hi.max(-1, keepdims=True)
    for _ in range(52):
        k = (left+right)*0.5
        y = np.clip(k,lo,hi)
        mask = (y*(y-k)).sum(-1,keepdims=True)>0
        left, right = np.where(mask,k,left), np.where(mask,right,k)
    y = np.clip((left+right)*0.5,lo,hi)
    f_hi = y.sum(-1)**2/(p.shape[-1]*(y*y).sum(-1))
    # A zero-support component stays zero under every multiplicative action.
    # Unit scale there gives a finite feasible representative before centering.
    u = np.log(np.divide(y, a, out=np.ones_like(a), where=a > 0))
    u -= u.mean(-1,keepdims=True)
    u *= np.minimum(1, bound/np.maximum(bound,np.max(np.abs(u),axis=-1,keepdims=True)))
    f_lo = np.maximum(conditional_fidelity(p, u),conditional_fidelity(p,centered_action(p,bound)))
    if (f_hi < f_lo - 1e-12).any():
        raise ArithmeticError("Reachability interval inverted")
    return f_lo,f_hi
