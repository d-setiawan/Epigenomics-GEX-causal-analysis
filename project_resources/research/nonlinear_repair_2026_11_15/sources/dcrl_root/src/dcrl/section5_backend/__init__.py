"""Backend used by the three independently validated Section 5 groups."""
from functools import lru_cache


@lru_cache(maxsize=1)
def estimator_class():
    from .local90_analytic_objective import AnalyticObjectiveEstimator
    from .local90_fast_gibbs import accelerated_gibbs_class
    from .local90_convergence_probe import convergence_probe_class
    return convergence_probe_class(
        accelerated_gibbs_class(AnalyticObjectiveEstimator), stop_rule=None)
