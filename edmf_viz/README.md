# EDMF Vizualizations
Tools for plotting vertical profiles from completed `TurbulenceConvection.jl` runs, including
    entrainment/detrainment rates, Pi groups, potential temperature, and specific humidity.

## Usage
To automatically plot existing `TurbulenceConvection.jl` runs, use plot_entr_script.py:

    python plot_profiles.py --tc_output_dir=DATA_DIR --save_figs_dir=SAVE_FIGS_DIR

where DATA_DIR is a directory containing TurbulenceConvection runs, and SAVE_FIGS_DIR is the directory to save the resulting figures.
