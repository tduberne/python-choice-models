import scipy.optimize as opt
import scipy.stats as stats
from scipy.misc import logsumexp
import pandas as pd
import numpy as np
import IPython.display as disp
import logging

log = logging.getLogger(__name__)

class Betas:
    def __init__(self, **start_values):
        # We allow specification with a value only: first convert all to Beta objects
        self.betas = {k: (b if isinstance(b, Beta) else Beta(b)) for k, b in start_values.items()}
        non_fixed = [(k, b) for k, b in self.betas.items() if not b.fixed]
        self.indices = {non_fixed[index][0]: index for index in range(len(non_fixed))}

        self.initial_array = np.array([beta.start_value for k, beta in non_fixed])
        self.names_non_fixed = [k for (k, v) in non_fixed]
        self.bounds = [b.bounds for k, b in non_fixed]
        
    def get(self, name, betas):
        # Return value from array if the parameter is not fixed, fixed value otherwise
        return betas[self.indices[name]] if name in self.indices else self.betas[name].start_value
    
    def to_dict(self, betas):
        return {name: betas[index] for (name, index) in self.indices.items()}
    
class Beta:
    def __init__(self, start_value,
                fixed=False,
                lower_bound=None,
                upper_bound=None):
        self.start_value = start_value
        self.fixed = fixed
        self.bounds = (lower_bound, upper_bound)
    
class EstimationResult:
    def __init__(self,
                 optimization_result,
                 estimates,
                 covar_matrix,
                 null_ll,
                 final_ll):
        self.optimization_result = optimization_result
        self.estimates = estimates
        self.covar_matrix = covar_matrix
        self.null_ll = null_ll
        self.final_ll = final_ll
        
        self.estimate_frame = pd.DataFrame(index=estimates.keys(),
                                           columns=['estimate', 'se', 't_stat_0', 'p_0', 't_stat_1', 'p_1'],
                                          dtype=float)

        for b in estimates.keys():
            self.estimate_frame.loc[b, 'estimate'] = estimates[b]
            self.estimate_frame.loc[b, 'se'] = np.sqrt(covar_matrix.loc[b,b])
 
        self.estimate_frame['t_stat_0'] = self.estimate_frame.estimate / self.estimate_frame.se
        self.estimate_frame['t_stat_1'] = (self.estimate_frame.estimate - 1) / self.estimate_frame.se
        
        self.estimate_frame['p_0'] = 2 * stats.norm.cdf(-np.abs(self.estimate_frame['t_stat_0']))
        self.estimate_frame['p_1'] = 2 * stats.norm.cdf(-np.abs(self.estimate_frame['t_stat_1']))

        self.goodness_fit = pd.DataFrame(index=['rho_sq', 'rho_bar_sq', "final_LL", 'null_LL', 'LL_ratio_test'], columns=['val'])
        self.goodness_fit.loc['null_LL', 'val'] = null_ll
        self.goodness_fit.loc['final_LL', 'val'] = final_ll
        self.goodness_fit.loc['rho_sq', 'val'] = 1 - final_ll / null_ll
        self.n_params = covar_matrix.shape[0]
        self.goodness_fit.loc['rho_bar_sq', 'val'] = 1 - (final_ll - self.n_params) / null_ll

        self.goodness_fit.loc['LL_ratio_test', 'val'] = -2 * (null_ll - final_ll)
    
    def _ipython_display_(self):
        disp.display_markdown('### Estimates:', raw=True)
        disp.display(self.estimate_frame)
        disp.display_markdown('### Goodness of fit:', raw=True)
        disp.display(self.goodness_fit)

def avail(x):
    """
    Helper function to take into account availability.
    
    Parameters
    ----------
    
    x : vector of a type that can be interpreted as booleans (typically integer or boolean)
    
    Returns
    -------
    
    Vector of the same length as x, with NaNs where x is False (0 in integer case), 0 elsewhere.
    This vector is thought to be added to the utility values, which has the effect
    of turning the utility to NaN if the alternative is not available.
    """
    return np.array([np.nan if not xi else 0 for xi in x])

def approx_jacobian(f, x, args=(), epsilon=0.00001):
    """
    Computes the gradient at each dimension of a vector-valued function.
    
    Based on scipy's "approx_fprime" function for scalar-valued functions.
    """
    f0 = f(*((x,) + args))
    
    grad = np.zeros((len(f0),len(x)), float)
    ei = np.zeros((len(x),), float)
    for i in range(len(x)):
        ei[i] = 1.0
        d = epsilon * ei
        fd = f(*((x + d,) + args))
        fp = (fd - f0) / d[i]
        
        for j in range(len(fp)):
            # could probably vectorized, but would require more testing
            grad[j, i] = fp[j]

        ei[i] = 0.0

    return grad

def likelihood_ratio_p_value(results, results_extended):
    """
    Compares the estimation results of two models, where the first model is a restricted version of
    the second one, that is, can be obtained from it by fixing the value of some parameters (typically to 0).

    Parameters
    ----------

    results: an EstimationResult object for the restricted model

    results_extended: an EstimationResult object for the extended model

    Returns
    -------

    A value between 0 and 1, corresponding to the probability tor reject the hypothesis that the
    extended model is "more correct" whereas it is not.
    """
    if results.n_params > results_extended.n_params:
        # allow the user to be confused...
        results, results_extended = results_extended, results

    test_stat = -2 * (results.final_ll - results_extended.final_ll)
    deg_freedom = results_extended.n_params - results.n_params

    return 1 - stats.chi2.cdf(test_stat, deg_freedom)
