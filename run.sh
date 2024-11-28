# Scripts to run to produce plots for assignment

# Part 1:
# - stats_assignment.py with n_pseudo=200 and mass range 105-155
# - stats_assignment.py with n_pseudo=200 and mass range 120-135


# Part 2:

python cls.py --n_pseudoexperiments_per_mass_bin 200 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 22  --plot_saving_folder "figures/cls_small_nsig22"
python cls.py --n_pseudoexperiments_per_mass_bin 200 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 45  --plot_saving_folder "figures/cls_small_nsig45"

python cls_enhanced.py --n_pseudoexperiments 400 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 22  --plot_saving_folder "figures/clsEnhanced_small_nsig22"
python cls_enhanced.py --n_pseudoexperiments 400 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 45  --plot_saving_folder "figures/clsEnhanced_small_nsig45"
python cls_enhanced.py --n_pseudoexperiments 400 --dataset data/toy_dataset.csv.npy --N_sig_expected 45  --plot_saving_folder "figures/clsEnhanced_large_nsig45"

