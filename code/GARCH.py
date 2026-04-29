import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t
from scipy.special import gammaln
from dataclasses import dataclass
from typing import Optional


# ── DATA CLASS FOR PARAMETERS ────────────────────────────────────────────────

@dataclass
class GARCHParams:
    mu:        float          # constant mean
    phi:       float          # AR(1) coefficient
    theta:     float          # MA(1) coefficient
    alpha:     float          # ARCH coefficient
    beta:      float          # GARCH coefficient
    omega:     float          # variance intercept (variance-targeted)
    sigma2_lr: float          # long-run variance (sample variance)
    nu:        Optional[float] = None   # degrees of freedom (t / skew-t)
    lam:       Optional[float] = None   # skewness parameter (skew-t only)

    def summary(self, annualise_factor: int = 4):
        print("═" * 45)
        print("  ARMA(1,1)-GARCH(1,1) Parameter Estimates")
        print("═" * 45)
        print(f"  Mean equation")
        print(f"    mu           = {self.mu:.6f}")
        print(f"    phi (AR1)    = {self.phi:.6f}")
        print(f"    theta (MA1)  = {self.theta:.6f}")
        print(f"  Variance equation")
        print(f"    omega        = {self.omega:.6f}  (variance-targeted)")
        print(f"    alpha        = {self.alpha:.6f}")
        print(f"    beta         = {self.beta:.6f}")
        print(f"    alpha+beta   = {self.alpha + self.beta:.6f}  "
              f"({'✓ stationary' if self.alpha + self.beta < 1 else '✗ NON-STATIONARY'})")
        print(f"  Long-run vol   = "
              f"{np.sqrt(self.sigma2_lr * annualise_factor) * 100:.2f}% (annualised)")
        if self.nu is not None:
            print(f"  Tail (nu)      = {self.nu:.4f}")
        if self.lam is not None:
            print(f"  Skew (lambda)  = {self.lam:.4f}")
        print("═" * 45)


# ── LOG-LIKELIHOOD FUNCTIONS ─────────────────────────────────────────────────

def _garch_variance_path(resid: np.ndarray, omega: float,
                         alpha: float, beta: float,
                         sigma2_lr: float) -> np.ndarray:
    """
    Compute the GARCH(1,1) conditional variance path.
    Initialised at the long-run variance (variance targeting start condition).
    """
    n = len(resid)
    sigma2 = np.empty(n)
    sigma2[0] = sigma2_lr                          # initialise at long-run variance
    for t in range(1, n):
        sigma2[t] = omega + alpha * resid[t-1]**2 + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], 1e-8)           # floor for numerical safety
    return sigma2


def _residuals(r: np.ndarray, mu: float,
               phi: float, theta: float) -> np.ndarray:
    """
    Compute ARMA(1,1) residuals:  eps_t = r_t - mu - phi*r_{t-1} - theta*eps_{t-1}
    """
    n = len(r)
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = r[t] - mu - phi * r[t-1] - theta * eps[t-1]
    return eps


def _nll_normal(params: np.ndarray, r: np.ndarray, sigma2_lr: float) -> float:
    """Gaussian ARMA-GARCH negative log-likelihood with variance targeting."""
    mu, phi, theta, alpha, beta = params

    # Stationarity and positivity guards — return large penalty if violated
    if not (abs(phi) < 1 and abs(theta) < 1):      return 1e10
    if not (alpha > 0 and beta > 0):                return 1e10
    if alpha + beta >= 0.9999:                      return 1e10

    omega = sigma2_lr * (1 - alpha - beta)          # variance targeting
    eps   = _residuals(r, mu, phi, theta)
    sig2  = _garch_variance_path(eps, omega, alpha, beta, sigma2_lr)

    # Gaussian log-likelihood: -0.5 * sum[ log(2pi) + log(sig2) + eps^2/sig2 ]
    nll = 0.5 * np.sum(np.log(2 * np.pi) + np.log(sig2) + eps**2 / sig2)
    return nll


def _nll_student_t(params: np.ndarray, r: np.ndarray, sigma2_lr: float) -> float:
    """Student-t ARMA-GARCH negative log-likelihood with variance targeting."""
    mu, phi, theta, alpha, beta, nu = params

    if not (abs(phi) < 1 and abs(theta) < 1):      return 1e10
    if not (alpha > 0 and beta > 0):                return 1e10
    if alpha + beta >= 0.9999:                      return 1e10
    if nu <= 2.01:                                  return 1e10   # variance must exist

    omega = sigma2_lr * (1 - alpha - beta)
    eps   = _residuals(r, mu, phi, theta)
    sig2  = _garch_variance_path(eps, omega, alpha, beta, sigma2_lr)
    sigma = np.sqrt(sig2)

    # Standardised t log-likelihood
    # Var(t_nu) = nu/(nu-2), so standardise innovations by sqrt(nu/(nu-2))
    scale = np.sqrt(nu / (nu - 2))
    z     = eps / (sigma * scale)      # z ~ t(nu) with unit variance
    
    log_const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    log_kern  = -((nu + 1) / 2) * np.log(1 + z**2 / (nu - 2))
    log_jacob = -np.log(sigma * scale)  # jacobian for sigma scaling

    nll = -np.sum(log_const + log_kern + log_jacob)
    return nll


def _nll_skewt(params: np.ndarray, r: np.ndarray, sigma2_lr: float) -> float:
    """
    Hansen (1994) Skewed-t ARMA-GARCH negative log-likelihood.
    Captures both fat tails (nu) and asymmetry (lam in (-1, 1)).
    """
    mu, phi, theta, alpha, beta, nu, lam = params

    if not (abs(phi) < 1 and abs(theta) < 1):       return 1e10
    if not (alpha > 0 and beta > 0):                 return 1e10
    if alpha + beta >= 0.9999:                       return 1e10
    if nu <= 2.01:                                   return 1e10
    if not (-0.999 < lam < 0.999):                   return 1e10

    omega = sigma2_lr * (1 - alpha - beta)
    eps   = _residuals(r, mu, phi, theta)
    sig2  = _garch_variance_path(eps, omega, alpha, beta, sigma2_lr)
    sigma = np.sqrt(sig2)
    z     = eps / sigma

    # Hansen (1994) constants
    c = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi * (nu - 2))
    a = 4 * lam * c * ((nu - 2) / (nu - 1))
    b = np.sqrt(1 + 3 * lam**2 - a**2)

    # Split domain: f(z) depends on sign of b*z + a
    bza = b * z + a
    ll  = np.empty(len(z))

    pos = bza >= 0
    # Right tail (z >= -a/b)
    ll[pos]  = (np.log(b) + np.log(c) - np.log(sigma[pos])
                - ((nu + 1) / 2)
                * np.log(1 + (bza[pos] / (1 + lam)) ** 2 / (nu - 2)))
    # Left tail (z < -a/b)
    ll[~pos] = (np.log(b) + np.log(c) - np.log(sigma[~pos])
                - ((nu + 1) / 2)
                * np.log(1 + (bza[~pos] / (1 - lam)) ** 2 / (nu - 2)))

    return -np.sum(ll)


# ── MAIN FITTING FUNCTION ────────────────────────────────────────────────────

def fit_garch(
    returns:   pd.Series,
    dist:      str = 'skewt',
    n_restarts: int = 5,
    seed:      int = 42
) -> GARCHParams:
    """
    Fit ARMA(1,1)-GARCH(1,1) with variance targeting via direct MLE.
    No arch library — pure numpy/scipy.

    Parameters
    ----------
    returns : pd.Series
        Return series (quarterly decimals, e.g. 0.05 = 5%)
    dist : str
        'normal', 't', or 'skewt'
    n_restarts : int
        Number of random restarts to escape local optima
    seed : int

    Returns
    -------
    GARCHParams dataclass
    """
    rng      = np.random.default_rng(seed)
    r        = returns.values.astype(float)
    sigma2_lr = r.var()

    dist_map = {
        'normal': (_nll_normal,    ['mu','phi','theta','alpha','beta']),
        't':      (_nll_student_t, ['mu','phi','theta','alpha','beta','nu']),
        'skewt':  (_nll_skewt,     ['mu','phi','theta','alpha','beta','nu','lam']),
    }
    if dist not in dist_map:
        raise ValueError(f"dist must be one of {list(dist_map.keys())}")

    nll_fn, param_names = dist_map[dist]

    # ── Starting values ──────────────────────────────────────────────────────
    def make_x0(random=False):
        if random:
            phi_0   = rng.uniform(-0.3, 0.3)
            theta_0 = rng.uniform(-0.3, 0.3)
            ab      = rng.uniform(0.70, 0.95)
            alpha_0 = rng.uniform(0.03, 0.25)
            beta_0  = ab - alpha_0
        else:
            phi_0, theta_0, alpha_0, beta_0 = 0.1, 0.05, 0.10, 0.85

        base = [r.mean(), phi_0, theta_0, alpha_0, beta_0]
        if dist == 't':     return base + [8.0]
        if dist == 'skewt': return base + [8.0, -0.1]
        return base

    # ── Bounds ───────────────────────────────────────────────────────────────
    base_bounds = [
        (-0.2, 0.2),    # mu
        (-0.95, 0.95),  # phi
        (-0.95, 0.95),  # theta
        (1e-6, 0.45),   # alpha
        (1e-6, 0.9999), # beta
    ]
    if dist == 't':     bounds = base_bounds + [(2.1, 50.0)]
    elif dist == 'skewt': bounds = base_bounds + [(2.1, 50.0), (-0.999, 0.999)]
    else:               bounds = base_bounds

    # ── alpha + beta < 1 constraint ──────────────────────────────────────────
    constraints = [{
        'type': 'ineq',
        'fun':  lambda x: 0.9999 - (x[3] + x[4])   # alpha + beta < 0.9999
    }]

    # ── Multi-start optimisation ─────────────────────────────────────────────
    best_result = None
    best_nll    = np.inf

    for i in range(n_restarts):
        x0 = make_x0(random=(i > 0))
        try:
            res = minimize(
                nll_fn,
                x0,
                args=(r, sigma2_lr),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-10, 'maxiter': 2000, 'disp': False}
            )
            if res.success and res.fun < best_nll:
                best_nll    = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("All optimisation restarts failed. "
                           "Check your return series for NaNs or extreme outliers.")

    x = best_result.x
    alpha, beta = x[3], x[4]
    omega = sigma2_lr * (1 - alpha - beta)

    params = GARCHParams(
        mu=x[0], phi=x[1], theta=x[2],
        alpha=alpha, beta=beta, omega=omega,
        sigma2_lr=sigma2_lr,
        nu=x[5]  if dist in ('t', 'skewt') else None,
        lam=x[6] if dist == 'skewt'        else None,
    )

    print(f"Converged in {best_result.nit} iterations  |  "
          f"NLL = {best_nll:.4f}  |  dist = {dist}")
    params.summary()
    return params


# ── INFORMATION CRITERIA ─────────────────────────────────────────────────────

def model_aic_bic(nll: float, n_params: int, n_obs: int) -> dict:
    aic = 2 * nll + 2 * n_params
    bic = 2 * nll + n_params * np.log(n_obs)
    return {'AIC': aic, 'BIC': bic, 'NLL': nll}


def select_distribution(returns: pd.Series, n_restarts: int = 5) -> str:
    """Fit all three distributions and select by BIC."""
    r = returns.values
    n = len(r)
    results = {}

    n_params_map = {'normal': 5, 't': 6, 'skewt': 7}

    print("Comparing distributions by BIC...\n")
    for dist in ['normal', 't', 'skewt']:
        try:
            params = fit_garch(returns, dist=dist, n_restarts=n_restarts)
            nll_fn = {'normal': _nll_normal,
                      't':      _nll_student_t,
                      'skewt':  _nll_skewt}[dist]
            x = [params.mu, params.phi, params.theta, params.alpha, params.beta]
            if params.nu  is not None: x.append(params.nu)
            if params.lam is not None: x.append(params.lam)
            nll = nll_fn(np.array(x), r, params.sigma2_lr)
            ic  = model_aic_bic(nll, n_params_map[dist], n)
            results[dist] = (params, ic)
            print(f"  {dist:<8}  AIC={ic['AIC']:.2f}  BIC={ic['BIC']:.2f}\n")
        except Exception as e:
            print(f"  {dist} failed: {e}")

    best = min(results, key=lambda d: results[d][1]['BIC'])
    print(f"  → Best distribution by BIC: {best.upper()}")
    return best, results[best][0]


# ── EXAMPLE USAGE ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)
    spy_returns = pd.Series(
        np.random.normal(0.015, 0.07, 80),
        name='SPY'
    )

    # Fit a single distribution
    params = fit_garch(spy_returns, dist='skewt', n_restarts=5)

    # Or select the best distribution automatically
    best_dist, best_params = select_distribution(spy_returns)