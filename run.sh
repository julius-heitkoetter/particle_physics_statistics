# Scripts to run to produce plots for assignment

if [ ! -d "logs" ]; then
  mkdir logs
fi

# Part 1:
# python stats_assignment_part1.py --n_pseudoexperiments 200 --mass_range_min 105 --mass_range_max 155 --dataset data/toy_dataset_smaller_peak.csv.npy --plot_saving_folder "figures/part1_small_mMin105_mMax155" > logs/part1_small_mMin105_mMax155 2>&1
# python stats_assignment_part1.py --n_pseudoexperiments 200 --mass_range_min 105 --mass_range_max 155 --dataset data/toy_dataset.csv.npy --plot_saving_folder "figures/part1_large_mMin105_mMax155" > logs/part1_large_mMin105_mMax155 2>&1
# python stats_assignment_part1.py --n_pseudoexperiments 200 --mass_range_min 120 --mass_range_max 135 --dataset data/toy_dataset.csv.npy --plot_saving_folder "figures/part1_large_mMin120_mMax135" > logs/part1_large_mMin120_mMax135 2>&1


# Part 2:

# python cls.py --n_pseudoexperiments_per_mass_bin 200 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 22  --plot_saving_folder "figures/cls_small_nsig22" > logs/cls_small_nsig22 2>&1
# python cls.py --n_pseudoexperiments_per_mass_bin 200 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 45  --plot_saving_folder "figures/cls_small_nsig45" > logs/cls_small_nsig45 2>&1

# python cls_enhanced.py --n_pseudoexperiments 400 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 22  --plot_saving_folder "figures/clsEnhanced_small_nsig22" > logs/clsEnhanced_small_nsig22 2>&1
# python cls_enhanced.py --n_pseudoexperiments 400 --dataset data/toy_dataset_smaller_peak.csv.npy --N_sig_expected 45  --plot_saving_folder "figures/clsEnhanced_small_nsig45" > logs/clsEnhanced_small_nsig45 2>&1
# python cls_enhanced.py --n_pseudoexperiments 400 --dataset data/toy_dataset.csv.npy --N_sig_expected 45  --plot_saving_folder "figures/clsEnhanced_large_nsig45" > logs/clsEnhanced_large_nsig45 2>&1

