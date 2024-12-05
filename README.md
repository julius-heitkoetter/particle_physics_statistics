# particle_physics_statistics

## Setup

Everything is run in python, so the only setup needed is to install the packages using pip. Using your favorite environment manager, create an environment and in it run:
```
pip install -r requirements.txt
```

If you have mac, it's a little more tricky to get things to work because the `zfit` package uses tensorflow, which needs to adapt to the M1 archetecture. The most straightforward way to do this is using conda, create an ARM environment:
```
# Create ARM conda environment
create_ARM_conda_environment () {

  # example usage: create_ARM_conda_environment myenv_x86 python=3.9
  CONDA_SUBDIR=osx-arm64 conda create -n $@
  conda activate $1

}
```
and then make sure tensorflow metal is installed.

## Code overview

Everything needed for part 1 of the assignment can be found in `stats_assignment_part1.py`. Most everything needed for part 2 of the assignment can be found in `cls.py`, with the exception of plotting in `cls_plotting.py` and the enhanced cls method with more mass bins, found in `cls_enhanced.py`. 

To run the code to produce the desired results, simply run `run.sh`. That's it!

If you don't feel like running it yourself, you can checkout the results in the `figures/` and `logs/` directories. Additionally, you can just get all the important information from the assignment writeup itself.