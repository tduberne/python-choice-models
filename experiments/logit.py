import scipy.optimize as opt
import scipy.stats as stats
import pandas as pd
import numpy as np


from typing import Mapping, TypeVar, Callable, List, Tuple

class Betas:
    def __init__(self, **start_values: Mapping[str, float]):
        l = list(start_values.items())
        self.indices = {l[index][0]: index for index in range(len(l))}
        self.initial_betas = np.array([val for (name, val) in l])
        self.names = [name for (name, _) in  l]
        
    def get(self, name: str, betas: np.array) -> float:
        return betas[self.indices[name]]
    
    def to_dict(self, betas: np.array) -> Mapping[str, float]:
        return {name: betas[index] for (name, index) in self.indices.items()}

class EstimationResult:
    def __init__(self,
                 estimates: Mapping[str, float],
                 covar_matrix: pd.DataFrame,
                 null_ll: float,
                 final_ll: float):
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



# Some aliases for type hints.
C = TypeVar('C')
Utilities = Mapping[C, Callable[[Betas, pd.DataFrame], float]]
Choices = pd.Series


def log_likelihood(betas: np.ndarray,
                       utilities: Utilities,
                       choices: Choices,
                       df: pd.DataFrame) -> float:
    if len(choices) != df.shape[0]:
        raise Exception('number of choices {} is different from number of observations {}'.format(len(choices), df.shape[0]))
    
    utility_values = pd.DataFrame({c: utilities[c](betas, df) for c in utilities.keys()})
    chosen_utility = utility_values.lookup(range(len(choices)), choices)
    
    # Numerical trick to avoid overflows in the sum of exponentials
    max_util = utility_values.max().max()
    logsums = max_util + np.log(np.exp(utility_values - max_util).sum(axis=1))
    loglikelihoods = chosen_utility - logsums
    
    return loglikelihoods

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

def score_matrix(betas: np.ndarray,
          utilities: Utilities,
          choices: Choices,
          df: pd.DataFrame) -> float:
    jac = approx_jacobian(log_likelihood, betas, args=(utilities, choices, df))
    
    N = jac.shape[1]
    K = len(betas)
    B = np.zeros((K,K), float)
    for i in range(N):
        scores = jac[..., i]
        out = scores.dot(scores.T)
        B += out / N
    
    return B

def estimate_logit(start_betas: Betas,
                   utilities: Utilities,
                   choice_vector: Choices,
                   dataset: pd.DataFrame) -> EstimationResult:
    result = opt.minimize(lambda x: -np.sum(log_likelihood(x, utilities, choice_vector, dataset)),
                          x0=start_betas.initial_betas)
    
    B = score_matrix(result.x, utilities, choice_vector, dataset)
    
    sandwich_est = result.hess_inv.dot(B).dot(result.hess_inv)
    covar_frame = pd.DataFrame(sandwich_est, index=start_betas.names, columns=start_betas.names)
    
    null_ll = np.sum(log_likelihood(result.x * 0, utilities, choice_vector, dataset))
    final_ll = -result.fun
    
    return EstimationResult(
        start_betas.to_dict(result.x),
        covar_frame,
        null_ll,
        final_ll)