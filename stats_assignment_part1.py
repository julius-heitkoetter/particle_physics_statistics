import os
os.environ['ZFIT_DISABLE_TF_WARNINGS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import zfit
from zfit import z
import hist
from hist import Hist
from tqdm import tqdm
import argparse

if __name__ == "__main__":

    np.random.seed(870987098)

    # Parameters used for fit, distinguished between FIXED values and
    # dummy values that serve as initial conditions for the fit
    M_HIGGS = 130 # mean of the gaussian at the Higgs mass DUMMY VALUE
    DETECTOR_RESOLUTION = 2 # width of the gaussian peak FIXED
    LAMBDA_PARAM = 1 # parameter of the exponential DUMMY VALUE
    FRAC_SIGNAL = 0.5 # fraction of signal over background DUMMY VALUE

    parser = argparse.ArgumentParser(description="Do calculations needed for part 1 of the statistic assignment")

    parser.add_argument(
        "--n_pseudoexperiments",
        type=int,
        default=200,
        help="Number of pseudoexperiments to run (default 200)."
    )
    parser.add_argument(
        "--mass_range_min",
        type=float,
        default=None,
        help="Minimum of the mass range to fit to, fit to entire data mass range if None (default: None)"
    )
    parser.add_argument(
        "--mass_range_max",
        type=float,
        default=None,
        help="Maximum of the mass range to fit to, fit to entire data mass range if None (default: None)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/toy_dataset_smaller_peak.csv.npy",
        help="Path to the dataset file, which contains the data for the analysis (default: data/toy_dataset_smaller_peak.csv.npy)."
    )
    parser.add_argument(
        "--plot_saving_folder",
        type=str,
        default="figures_part1",
        help="Path to save the figures which are the output of the script (default: figures_CLsMethod)"
    )

    args = parser.parse_args()
    # Print the parsed arguments
    print("Calculations needed for part 1 of the statistic assignment")
    print("Arguments:")
    print(f"  n_pseudoexperiments: {args.n_pseudoexperiments}")
    print(f"  mass_range_min: {args.mass_range_min}")
    print(f"  mass_range_max: {args.mass_range_max}")
    print(f"  dataset: {args.dataset}")
    print(f"  plot_saving_folder: {args.plot_saving_folder}")

    # Make the output folder
    if not os.path.exists(args.plot_saving_folder):
        os.makedirs(args.plot_saving_folder)

    # Get the input data
    data = np.load(args.dataset)
    m_range = [np.min(data)-1, np.max(data)+1] # range of the data
    if args.mass_range_min and args.mass_range_min > m_range[0]: # Adjust the mass range to accompany changes in script input parameters.
        m_range[0] = args.mass_range_min
    if args.mass_range_max and args.mass_range_max < m_range[1]:
        m_range[1] = args.mass_range_max
    #data = data[data >= m_range[0]]
    #data = data[data <= m_range[1]]
    x_obs = zfit.Space("x", limits=m_range) # convert m_range into a zfit space
    zfit_data = zfit.Data(data, obs=x_obs) # convert data to zfit type


    # Define models to be used for fitting
    exp_lambda = zfit.Parameter("lambda", LAMBDA_PARAM) 
    exp_model = zfit.pdf.Exponential(lam=exp_lambda, obs=zfit.Space("x", limits=m_range))
    gauss_mean = zfit.Parameter("mean", M_HIGGS, lower=m_range[0], upper=m_range[1])  
    gauss_sigma = zfit.Parameter("sigma", DETECTOR_RESOLUTION, floating=False) # we fix the width of the Gaussian peak! i.e. this will not be fitted
    gauss_model = zfit.pdf.Gauss(mu=gauss_mean, sigma=gauss_sigma, obs=zfit.Space("x", limits=m_range))
    frac_exp = zfit.Parameter("frac_exp", 1-FRAC_SIGNAL, 1-2*FRAC_SIGNAL, 1) 
    combined_model = zfit.pdf.SumPDF([exp_model, gauss_model], fracs=frac_exp, obs=x_obs)

    # Fit the data and get liklihoods
    minimizer = zfit.minimize.Minuit()
    alt_nll = zfit.loss.UnbinnedNLL(model=combined_model, data=zfit_data)
    alt_result = minimizer.minimize(alt_nll)
    null_nll = zfit.loss.UnbinnedNLL(model=exp_model, data=zfit_data)
    null_result = minimizer.minimize(null_nll)
    print(f"Alternative hypothosis NLL: {alt_nll.value()}")
    print(f"Null hypothosis NLL: {null_nll.value()}")
    print(f"N sig events found:  {(1-frac_exp) * len(zfit_data)}")

    # calculate the profile likelihood
    profile_likelihood = 2   * (null_nll.value() - alt_nll.value())
    print(f"Profile Likelihood is :", profile_likelihood)

    # Compare with a Chi^2 with 2 dof to get p value
    p_value = stats.chi2.sf(profile_likelihood, df=2)
    print(f"P-value: {p_value}")

    # Calculate the trial factor:
    Z_fix = np.sqrt(1-p_value)
    N = (m_range[1] - m_range[0])/DETECTOR_RESOLUTION
    trial_factor = 1 + (np.sqrt(np.pi/2) * Z_fix * N)
    print(f"Trial factor is: ", trial_factor)

    #Create 200 pseudoexperiments under the null hypothosis
    n_data_points = len(zfit_data)
    null_profile_likelihoods = []
    print(">>> STARTING GENERATING TOYS <<<")
    for _ in tqdm(range(args.n_pseudoexperiments)):

        # Create pseudodata from the exp model that was fit for background only
        pseudo_data = exp_model.sample(n=n_data_points)
        
        # Compute NLLs
        alt_nll = zfit.loss.UnbinnedNLL(model=combined_model, data=pseudo_data)
        alt_result = minimizer.minimize(alt_nll)
        null_nll = zfit.loss.UnbinnedNLL(model=exp_model, data=pseudo_data)
        null_result = minimizer.minimize(null_nll)
        
        # Calculate profile liklihood and store it
        null_profile_likelihood = 2 * (null_nll.value() - alt_nll.value())
        null_profile_likelihoods.append(null_profile_likelihood)

    pseudo_experiment_p_value = np.mean(null_profile_likelihoods > profile_likelihood)
    print(f"P-value gotten from pseudoexperiments: {pseudo_experiment_p_value}")

    # Calculate trial factor from the toys
    empirical_Z_toys = stats.norm.ppf(1 - pseudo_experiment_p_value)
    empirical_Z_fix = stats.norm.ppf(1 - p_value)
    trial_factor_from_toys = empirical_Z_toys / empirical_Z_fix
    print(f"Trial factor from toys: {trial_factor_from_toys}")


    #####################################
    #############   PLOTS   #############
    #####################################

    fig = plt.figure(figsize=(5,5))
    ax = fig.subplots()

    # define a histogram
    hist = Hist.new.Reg(30, m_range[0], m_range[1]).Weight()
    hist.fill(data[:,0])

    # and plot it
    ax.errorbar(hist.axes[0].centers, hist.values(), np.sqrt(hist.variances()), color='black', capsize=2, linestyle='', marker='o', markersize=3, label='Data')
    ax.set_xlabel(r"$\mathrm{m}_{\gamma\gamma}$ [GeV]")
    ax.set_ylabel('Events')
    ax.legend(fontsize='small')
    plt.savefig(os.path.join(args.plot_saving_folder,'histogram_of_data.pdf'))
    plt.close()


    # Create the Chi^2 distribution with 2 degrees of freedom
    x = np.linspace(0, max(null_profile_likelihoods) + 5, 1000)
    chi2_pdf = stats.chi2.pdf(x, df=2)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.hist(null_profile_likelihoods, bins=30, density=True, alpha=0.6, label='Null Profile Likelihoods')
    plt.plot(x, chi2_pdf, label=r'$\chi^2$ PDF with 2 DOF', color='red', linestyle='--')

    # Add vertical line for observed profile likelihood
    plt.axvline(profile_likelihood, color='blue', linestyle='-', label=f'Observed Profile Likelihood = {profile_likelihood:.2f}')

    # Shade region above observed profile likelihood
    x_shade = np.linspace(profile_likelihood, max(x), 1000)
    y_shade = stats.chi2.pdf(x_shade, df=2)
    plt.fill_between(x_shade, y_shade, alpha=0.3, color='blue', label=f'Shaded region: Wilks p = {p_value:.3f}\nPseudoexperiments p = {pseudo_experiment_p_value:.3f}')

    plt.text(0.05, 0.85, f'Trial Factor (Analytic): {trial_factor:.2f}\n'
                        f'Trial Factor (From Toys): {trial_factor_from_toys:.2f}',
            transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Labels and legend
    plt.title('Profile Likelihood Distribution vs Chi^2 Distribution', fontsize=16)
    plt.xlabel('Profile Likelihood', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(alpha=0.4)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_saving_folder,"profile_liklihoods.pdf"))
    plt.close()