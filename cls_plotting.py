import matplotlib.pyplot as plt
from hist import Hist
import numpy as np

def plot_cls_profile_likelihood_distribution(m, sb_profile_likelihoods, b_profile_likelihoods, observed_profile_likelihood,  p5_threshold, power_at_p5):

    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(b_profile_likelihoods, bins=30, alpha=0.7, label='Background only', color='blue', density=True)
    plt.hist(sb_profile_likelihoods, bins=30, alpha=0.7, label='Signal + Background', color='orange', density=True)

    # Add observed profile likelihood as a vertical line
    plt.axvline(observed_profile_likelihood, color='red', linestyle='--', label='Observed Profile Likelihood')
    # Add observed profile likelihood as a vertical line
    plt.axvline(p5_threshold, color='green', linestyle='--', label=f'p=0.05 threshold value, with power = {power_at_p5:.2f}')
    
    # Add labels and title
    plt.title(f"Profile Likelihood Distributions for Mass Bin {m} GeV", fontsize=16)
    plt.xlabel("Profile Likelihood", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    
    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/cls_profile_liklihood_distribution_m={m}.pdf")

def plot_cls_sb_profile_likelihoods_mass_binned(sb_profile_likelihoods_mass_binned):

    plt.figure(figsize=(10, 6))

    # Plot histograms
    for m, sb_profile_likelihoods in sb_profile_likelihoods_mass_binned.items():
        plt.hist(sb_profile_likelihoods, bins=30, alpha=0.6, label=f'Signal + Background for m={m}', density=True)
    
    # Add labels and title
    plt.title(f"S+B Profile Likelihood Distributions", fontsize=16)
    plt.xlabel("Profile Likelihood", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    
    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/cls_sb_profile_likelihoods_mass_binned.pdf")

def plot_cls_b_profile_likelihoods(b_profile_likelihoods):

    plt.figure(figsize=(10, 6))

    # Plot histogram
    plt.hist(b_profile_likelihoods, bins=30, alpha=0.6, label=f'Background Only', density=True)
    
    # Add labels and title
    plt.title(f"B Only Profile Likelihood Distributions", fontsize=16)
    plt.xlabel("Profile Likelihood", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    
    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/cls_b_profile_likelihoods.pdf")


def plot_fit(data, m_range, observed_combined_model_null, observed_combined_model_alt, m_label):

    # Define histogram
    fig = plt.figure(figsize=(7, 7))
    ax = fig.subplots()

    # Fill histogram with data
    hist = Hist.new.Reg(30, m_range[0], m_range[1], name="x").Weight()
    hist.fill(data[:, 0])

    # Plot the histogram with error bars
    ax.errorbar(
        hist.axes[0].centers, 
        hist.values(), 
        np.sqrt(hist.variances()), 
        color='black', 
        capsize=2, 
        linestyle='', 
        marker='o', 
        markersize=3, 
        label='Data'
    )

    # Generate a range of x values for the fit lines
    x_vals = np.linspace(m_range[0], m_range[1], 1000)

    # Evaluate the null hypothesis fit
    y_null = observed_combined_model_null.pdf(x_vals)

    # Evaluate the alternative hypothesis fit
    y_alt = observed_combined_model_alt.pdf(x_vals)

    # Scale the PDFs to the histogram's total count
    scaling_factor = np.sum(hist.values()) * (m_range[1] - m_range[0]) / 30
    y_null_scaled = y_null * scaling_factor
    y_alt_scaled = y_alt * scaling_factor

    # Plot the fit lines
    ax.plot(x_vals, y_null_scaled, label='Null Hypothesis Fit', color='blue', linewidth=2)
    ax.plot(x_vals, y_alt_scaled, label='Alternative Hypothesis Fit', color='red', linewidth=2)

    # Labeling and saving the plot
    ax.set_xlabel(r"$\mathrm{m}_{\gamma\gamma}$ [GeV]")
    ax.set_ylabel('Events')
    ax.set_title(f"Observed Data w/ fit constrained to m={m_label}")
    ax.legend(fontsize='small')
    plt.savefig(f'figures/histogram_with_fits_m={m_label}.pdf')

def plot_cls(cls_dict):

    # Extract keys and values for plotting
    masses = list(cls_dict.keys())
    cls = list(cls_dict.values())

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(masses, cls, label='Obs.', color='black', linestyle='-', marker='o')

    # Add horizontal lines
    plt.axhline(y=0.05, color='red', linestyle='--', label='95%')
    plt.axhline(y=0.01, color='red', linestyle='-.', label='99%')
    plt.axhline(y=1, color='black', linestyle='-', label='CLs = 1')

    # Log scale for y-axis
    plt.yscale('log')
    plt.ylim(1e-8, 1e1)  # Match y-axis limits from the reference image

    # Add labels and legend
    plt.xlabel(r'$m_H$ [GeV]', fontsize=14)
    plt.ylabel('CLs', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')

    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add ticks for better alignment
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the plot
    plt.tight_layout()
    plt.savefig("cls_plot.pdf")

if __name__ == "__main__":
    cls_values = {
        100: 0.01,
        150: 0.02,
        200: 1e-3,
        250: 1e-5,
        300: 1e-6,
        350: 1e-4,
        400: 1e-3,
        450: 0.02,
        500: 0.1,
        550: 0.5,
        600: 1.0,
    }  

    plot_cls(cls_values)