from pyext.src.typedefs import Index as Index, PRNGKey as PRNGKey
from typing import Union

def testdef_get_trivial_model(n_samples, n_inter): ...
def trvial_is_get_invariants_jittable(n_samples, n_inter) -> None: ...
def trivial_is_s_rv_jittable(n_samples, n_inter) -> None: ...
def trivial_is_T_jittable(n_samples, n_inter) -> None: ...
def trivial_is_get_log_score_jittable(n_samples, n_inter) -> None: ...
def not_ones_trivial(n_samples, n_inter) -> None: ...
def specialize_model_to_sampling_trivial(n_samples: int, n_inter: int, decimals): ...
def sample_trivial(n_samples: int, n_inter: int, decimal_tolerance: int): ...
def negative_sample_trivial(n_samples: int, n_inter: int, rseed1: int, rseed2: int): ...
def nsteps_mh__g(mu: float, sigma: float, rseed: Union[float, int]): ...
def nsteps_mh__g_accuracy(mu, cv) -> None: ...
def apply_normal_context_to_sample(mu: float, sigma: float, n_mh_steps: int, n_samples: int, n_inter: int, rseed): ...
def f0_pdf__j(mu, sig) -> None: ...
def fn_pdf__j(mu, sig) -> None: ...
def fj_pdf__g(mu, sig) -> None: ...
def T_nsteps__unorm2unorm__p(mu, sig) -> None: ...
def test_T_nsteps_mh__g(rseed, x) -> None: ...
def do_ais(mu, sigma) -> None: ...