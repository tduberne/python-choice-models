import scipy.optimize as opt
import scipy.stats as stats
from scipy.misc import logsumexp
import pandas as pd
import numpy as np
import logging
import loglikelihood_utils as ll

log = logging.getLogger(__name__)

def log_likelihood(betas,
                       utilities,
                       choices,
                       df):
    log.debug('evaluating betas {}'.format(betas))
    if len(choices) != df.shape[0]:
        raise Exception('number of choices {} is different from number of observations {}'.format(len(choices), df.shape[0]))
 
    # transform missing utilities to infinitely negative utilities (handled more consistently than NaN as missings)
    utility_values = pd.DataFrame({c: utilities[c](betas, df) for c in utilities.keys()}).fillna(-np.inf)
    chosen_utility = utility_values.lookup(utility_values.index, choices)
    
    # Numerical trick to avoid overflows in the sum of exponentials
    logsums = logsumexp(utility_values, axis=1)
    loglikelihoods = chosen_utility - logsums

    log.debug('LL={}'.format(np.nansum(loglikelihoods)))
    return loglikelihoods

def estimate(start_betas,
                   utilities,
                   choice_vector,
                   dataset):
    result = opt.minimize(lambda x: -np.nansum(log_likelihood(x, utilities, choice_vector, dataset)),
                          x0=start_betas.initial_array,
                          bounds=start_betas.bounds,
                          options={'disp': True})
    
    B = score_matrix(result.x, utilities, choice_vector, dataset)
    
    sandwich_est = result.hess_inv.dot(B).dot(result.hess_inv.todense())
    covar_frame = pd.DataFrame(sandwich_est, index=start_betas.names_non_fixed, columns=start_betas.names_non_fixed)
    
    null_ll = np.nansum(log_likelihood(result.x * 0, utilities, choice_vector, dataset))
    final_ll = -result.fun
    
    return ll.EstimationResult(
        start_betas.to_dict(result.x),
        covar_frame,
        null_ll,
        final_ll)

# Could easily be moved to utils
def score_matrix(betas,
          utilities,
          choices,
          df):
    jac = ll.approx_jacobian(log_likelihood, betas, args=(utilities, choices, df))

    N = jac.shape[0]
    K = len(betas)
    B = np.zeros((K,K), float)
    for i in range(N):
        scores = jac[i, ...]
        # ill-specified dataset might have rows where no option is available,
        # resulting in NaN score. We want to handle such cases gracefully.
        if np.all(np.isnan(scores)):
            continue
        out = scores.T.dot(scores)
        B += out / N
    
    return B
