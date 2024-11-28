# Scripts to run to produce plots for assignment

if [ ! -d "logs" ]; then
  mkdir logs
fi

# Part 1:
# - stats_assignment.py with n_pseudo=200 and mass range 105-155
# - stats_assignment.py with n_pseudo=200 and mass range 120-135


# Part 2:

python cls.py --n_pseudoexperiments_per_mass_bin 200 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 22  --plot_saving_folder "figures/cls_small_nsig22" > logs/cls_small_nsig22 2>&1
python cls.py --n_pseudoexperiments_per_mass_bin 200 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 45  --plot_saving_folder "figures/cls_small_nsig45" > logs/cls_small_nsig45 2>&1

python cls_enhanced.py --n_pseudoexperiments 400 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 22  --plot_saving_folder "figures/clsEnhanced_small_nsig22" > logs/clsEnhanced_small_nsig22 2>&1
python cls_enhanced.py --n_pseudoexperiments 400 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 45  --plot_saving_folder "figures/clsEnhanced_small_nsig45" > logs/clsEnhanced_small_nsig45 2>&1
python cls_enhanced.py --n_pseudoexperiments 400 --dataset data/toy_dataset.csv.npy --N_sig_expected 45  --plot_saving_folder "figures/clsEnhanced_large_nsig45" > logs/clsEnhanced_large_nsig45 2>&1

