# IDEA: the S+B is the null hypothosis, which should just obey a chi2 with 1 dof
#       and the B only should be independent of the mass parameter. Therefore, we
#       should be able to just generate one S+B profile liklihood distribution from
#       Wilk's theorm and generate only 1 set of toys for B only hypothosis. Then, all
#       that is left to do is make many observed test statistics using the real data.


import os
os.environ['ZFIT_DISABLE_TF_WARNINGS'] = '1'

import numpy as np
import scipy.stats as stats
import zfit
from zfit import z
import hist
from hist import Hist
from tqdm import tqdm

from cls_plotting import plot_fit, plot_cls

np.random.seed(870987098)

n_pseudoexperiments = 200
mass_bin_width = 0.5
N_sig_expected = 45 #21.519628102502676

# Parameters used for fit, distinguished between FIXED values and
# dummy values that serve as initial conditions for the fit
M_HIGGS = 130 # mean of the gaussian at the Higgs mass DUMMY VALUE
DETECTOR_RESOLUTION = 2 # width of the gaussian peak FIXED
LAMBDA_PARAM = 1 # parameter of the exponential DUMMY VALUE
FRAC_SIGNAL = 0.5 # fraction of signal over background DUMMY VALUE

# Get the input data
data = np.load("data/toy_dataset.csv.npy")
m_range = (110, 150) # range of the data
m_bins = np.arange(m_range[0], m_range[1], mass_bin_width)
x_obs = zfit.Space("x", limits=m_range) # convert m_range into a zfit space
zfit_data = zfit.Data(data, obs=x_obs) # convert data to zfit type
n_data_points = len(zfit_data)

observed_profile_likelihoods_mass_binned = dict()
print(">>> STARTING OBSERVED DATA FITTING <<<")
for m in tqdm(m_bins):

    # Define models to be used for fitting data 
    observed_exp_lambda = zfit.Parameter("lambda", LAMBDA_PARAM) 
    observed_exp_model = zfit.pdf.Exponential(lam=observed_exp_lambda, obs=zfit.Space("x", limits=m_range))
    observed_gauss_mean = zfit.Parameter("mean", m, floating=False)  
    observed_gauss_sigma = zfit.Parameter("sigma", DETECTOR_RESOLUTION, floating=False) # we fix the width of the Gaussian peak! i.e. this will not be fitted
    observed_gauss_model = zfit.pdf.Gauss(mu=observed_gauss_mean, sigma=observed_gauss_sigma, obs=zfit.Space("x", limits=m_range))
    observed_frac_exp = zfit.Parameter("frac_exp", 1-FRAC_SIGNAL, 1-2*FRAC_SIGNAL, 1) 
    observed_frac_exp_fixed = zfit.Parameter("frac_exp", 1 - N_sig_expected/n_data_points, floating=False) 
    observed_combined_model_alt = zfit.pdf.SumPDF([observed_exp_model, observed_gauss_model], fracs=observed_frac_exp, obs=x_obs)
    observed_combined_model_null = zfit.pdf.SumPDF([observed_exp_model, observed_gauss_model], fracs=observed_frac_exp_fixed, obs=x_obs)

    # Fit the data and get bckg model.
    minimizer = zfit.minimize.Minuit()
    observed_alt_nll = zfit.loss.UnbinnedNLL(model=observed_combined_model_alt, data=zfit_data)
    observed_alt_result = minimizer.minimize(observed_alt_nll)
    observed_null_nll = zfit.loss.UnbinnedNLL(model=observed_combined_model_null, data=zfit_data)
    observed_null_result = minimizer.minimize(observed_null_nll)

    # calculate the observed profile likelihood and store it
    observed_profile_likelihood = 2   * (observed_null_nll.value() - observed_alt_nll.value())
    observed_profile_likelihoods_mass_binned[m] = observed_profile_likelihood

    # Plot the fit
    plot_fit(data, m_range, observed_combined_model_null, observed_combined_model_alt, m_label=m, data_type="Observed")

print(">>> STARTING GENERATING BACKGROUND TOYS <<<")
b_profile_likelihoods = []
for _ in tqdm(range(n_pseudoexperiments)):

    # Create pseudodata from the exp model that was fit for background only
    toy_data = observed_exp_model.sample(n=n_data_points)

    # Define models to be used for fitting and generating data
    toy_exp_lambda = zfit.Parameter("lambda", observed_exp_lambda.value()) # Get the lambda value from the data fit
    toy_exp_model = zfit.pdf.Exponential(lam=toy_exp_lambda, obs=zfit.Space("x", limits=m_range))
    toy_gauss_mean = zfit.Parameter("mean", M_HIGGS, floating=False)  # we fix the location of the Gaussian peak! i.e. this will not be fitted
    toy_gauss_sigma = zfit.Parameter("sigma", DETECTOR_RESOLUTION, floating=False) # we fix the width of the Gaussian peak! i.e. this will not be fitted
    toy_gauss_model = zfit.pdf.Gauss(mu=toy_gauss_mean, sigma=toy_gauss_sigma, obs=zfit.Space("x", limits=m_range))
    toy_frac_exp = zfit.Parameter("frac_exp", 1 - N_sig_expected/n_data_points, 1-2*FRAC_SIGNAL, 1) # Get the fraction value from the expected number of signal events
    toy_frac_exp_fixed = zfit.Parameter("frac_exp", 1 - N_sig_expected/n_data_points, floating=False)
    toy_combined_model_alt = zfit.pdf.SumPDF([toy_exp_model, toy_gauss_model], fracs=toy_frac_exp, obs=x_obs) # Under the alt hypothosis, the signal fraction is left floating
    toy_combined_model_null = zfit.pdf.SumPDF([toy_exp_model, toy_gauss_model], fracs=toy_frac_exp_fixed, obs=x_obs) # Under the null hypothosis, we propose signal at a certain strength 

    # Fit backgroud only model and get NLL
    toy_alt_nll = zfit.loss.UnbinnedNLL(model=toy_combined_model_alt, data=toy_data)
    toy_alt_result = minimizer.minimize(toy_alt_nll)
    toy_null_nll = zfit.loss.UnbinnedNLL(model=toy_combined_model_null, data=toy_data)
    toy_null_result = minimizer.minimize(toy_null_nll)

    # Calculate the background only profile liklihood and store it
    toy_profile_likelihood = 2   * (toy_null_nll.value() - toy_alt_nll.value())
    b_profile_likelihoods.append(toy_profile_likelihood)

# Generate the S+B profile likelihoods using Wilks
x = np.linspace(1e-3, 2*observed_profile_likelihood.numpy() + 5, 1000)
sb_profile_likelihoods = stats.chi2.rvs(df=1, size=n_pseudoexperiments * 100)# Wilks allows us to assume chi2 with 1 dof for the s+b (null) hypothosis

# Generate data for cls plot
cls_values = dict()
for m in observed_profile_likelihoods_mass_binned.keys():
    power = np.mean(b_profile_likelihoods > observed_profile_likelihoods_mass_binned[m])
    p_value = np.mean(sb_profile_likelihoods > observed_profile_likelihoods_mass_binned[m])
    cls_values[m] = p_value/power

plot_cls(cls_values)