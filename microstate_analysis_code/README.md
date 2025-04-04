# Charge microstate analysis with weighted correlation

  * Stand-alone charge microstate analysis with correlation.
  * Can be used without the mcce program & its codebase.

### Environment:
Before using `cms_analysis_wc.py` or the related notebook, `run_cms_analysis.ipynb`, ensure
that you have an environment setup for python 3.10 with the following packages:
```
matplotlib
numpy
pandas
seaborn
scipy
```

### Command line call example:
Parameter passing via a file:
```
  python cms_analysis_wc.py params.crgms  
```

### running the module outside of the repo:
 1. Copy the module to your user/bin folder
 2. Copy one of the '.crgms' input parameter file into your folder of interest
 3. Amend the file to your location & needs
 4. Run: `cms_analysis_wc.py your.crgms`

### Help option output: `./cms_analysis_wc.py --help`
```
usage:
CALL EXAMPLE:
  python cms_analysis_wc.py params.crgms

OR, if you copied cms_analysis_wc.py in your user/bin folder:
  cms_analysis_wc.py params.crgms

The input parameter file must be found at the location where
the module is called.

Note:
  If you add this line in your input parameter file:
     list_head3_ionizables = true
  the program will list the resids in head3.lst and exit;
  The list or a portion thereof can then be used as values to
  the 'correl_resids' identifier.

cms_analysis_wc.py ::
Stand-alone charge microstate analysis with correlation.
Can be used without the mcce program & its codebase.
The only command line option is the parameters file.

positional arguments:
  params_file  The input parameters file.

options:
  -h, --help   show this help message and exit
```

### Input parameter file example: param.crgms:
```
# Input file to perform charge microstates analysis with weighted correlation.
# Original filename: params.crgms :: Can be changed as long as the format is retained.
#
# Data lines:
#   All non-blank, non-commented lines will be parsed for data with this format: <identifier> = <value(s)>.
#   Spaces around the equal sign are not required.
#   Quotes around strings values, or items in a list or tuple are not required.

# Comments: Not on a data line.

# Data entry format:
#   * Values as lists (single line); residue_kinds and correl_resids.
#     Example:
#       residue_kinds = [ ASP, GLU, ARG, HIS, LYS, TYR, NTR, 'CTR']
#       correl_resids = [ GLUA0035_,  GLUA0055_, HISA0015_ ASPA0119_  ]

#   * Values as 2-tuple, e.g.: .fig_size, .bounds: comma separated values within parentheses: (20, 8), (20,8).
#   * All identifiers for plots can have attributes given with '.<attribute>, e.g.: energy_histogram.fig_size = (8,8).
#    - Valid attributes: .save_name, .fig_size, .title (and .bounds for the 'charge_histogramN' identifier).
#
# INPUT/OUTPUT PATHS SELECTION
# ========================================================================================
  # mcce_dir :: The directory of a MCCE run containing a ms_out subfolder :: Default: current directory.
  mcce_dir = 4lzt

  # output_dir :: The mcce_dir subdirectory for saving the analysis files :: Default: crgms_corr.
  #output_dir = crgms_corr

  # To return multi-line string of resids ready to use as values of the 'correl_resids' identifier.
  # Default is False. If True, the list will be output to screen and the program will exit.
  # Valid values: false, true, False, True.
  list_head3_ionizables = False

  # Files names:

  # free and fixed residues charges:
  all_res_crg_csv = all_res_crg_status.csv

  # residue of interest data (not renamed):
  res_of_interest_data_csv = crg_count_res_of_interest.csv


# MICROSTATES FILE SELECTION
# ========================================================================================
# The 'msout file' that resides in mcce_dir/ms_out is named using the ph and eh points of
# the titration, e.g.: pH7eH0ms.txt, or pH7.5eH30ms.txt.
#
  msout_file = pH7eH0ms.txt

# occupancy threshold used for returning unique charge ms:
  min_occ = 0.02

# n_top: number of unique charge ms to return
  n_top = 5


# RESIDUES SELECTION
# ========================================================================================
  # residue_kinds :: 3-letter code of residues of interest for filtering the microstates collection.
  # Default: If the 'residue_kinds line' is absent, commented out, or its value is an empty list,
  #          all ionizable residues are used for filtering.
  # Note: order or duplicates do not matter.
  #
  residue_kinds = [ASP, HEM, PL9, GLU, HIS, TYR, NTR, CTR]

  # correl_resids :: list of conformer ids (among residue_kinds) to test for correlation.
  #  Format: 3-letter resid + chain letter + 4 digit seqnum + _
  correl_resids = [  GLUA0035_ GLUA0055_,  HISA0015_ ASPA0119_ ]


# CORRELATION PARAMETERS
# ========================================================================================
  # corr_method: either pearson or spearman:
  corr_method = pearson
  corr_cutoff = 0

  # Change the number of clusters to pass to `cluster_corr_matrix` from default 5:
  #n_clusters = 9


# FIGURES PARAMETERS & ATTRIBUTES
# ========================================================================================

# Global flag to indicate whether to show each plot:
  fig_show = False

# Plots:

  # Filenames:
  energy_histogram.save_name = enthalpy_dist.png
  corr_heatmap.save_name = corr.png

  # Titles:

  # Add a title attribute to the correlation heatmap if needed (default is no title):
  #corr_heatmap.title = Loaded

  # Figure sizes
  energy_histogram.fig_size = (8,8)
  corr_heatmap.fig_size = (20, 8)

  # Charge microstates histograms:
  #   Multiple histograms can be created using the 'charge_histogram' identifier,
  #   which can end with integers or characters for clarity.
  #   Examples for plots with bounds vz Emin, Eaver & Emax:

  # charge_histogram0: Always included, creates all_crg_count_res: no filtering, value = (None, None):
  charge_histogram0.bounds = (None, None)
  charge_histogram0.title = Protonation MS Energy
  charge_histogram0.save_name = crgms_logcount_vs_E.png

  charge_histogram1.bounds = (Emin, Emin + 1.36)
  charge_histogram1.title = Protonation MS Energy within 1.36 kcal/mol of Lowest
  charge_histogram1.save_name = crgms_logcount_vs_lowestE.png

  charge_histogram2.bounds = (Eaver - 0.68, Eaver + 0.68)
  charge_histogram2.title = Protonation MS Energy within 0.5 pH (0.68 kcal/mol) of Average
  charge_histogram2.save_name = crgms_logcount_vs_averE.png

  charge_histogram3.bounds = (Emax - 1.36, Emax)
  charge_histogram3.title = Protonation MS Energy within 1.36 kcal/mol of Highest
  charge_histogram3.save_name = crgms_logcount_vs_highestE.png

  # example with free bounds:
  charge_histogramB.bounds = (-5, 5)
  charge_histogramB.title = Protonation MS Energy in (-5, 5) kcal/mol range
  charge_histogramB.save_name = crgms_logcount_vs_rangeE.png
```
