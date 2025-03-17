#!/usr/bin/env python

"""
Module: cms_analysis_wc.py

  Stand-alone charge microstate analysis with correlation.
  Can be used without the mcce program & its codebase.
  Can be obtain from the codebase using wget
  TBD: give MCCE4 repo path once finalized.

COMMAND LINE CALL EXAMPLE. Parameter passing via a file:
  python cms_analysis_wc.py params.crgms  
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import defaultdict
from copy import deepcopy
import logging
import math
import operator
from pathlib import Path
import re
import string
import sys
from typing import List, Tuple, Union
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.ioff()
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import skewnorm, rankdata


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(funcName)s:\n\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


ph2Kcal = 1.364
half_ph = ph2Kcal / 2


IONIZABLES = ["ASP", "GLU", "ARG", "HIS", "LYS", "CYS", "TYR", "NTR", "CTR"]
ACIDIC_RES = ["ASP", "GLU"]
BASIC_RES = ["ARG", "HIS", "LYS"]
POLAR_RES = ["CYS", "TYR"]
CORR_METHODS = ["pearson", "spearman"]


# Mapping of 3-letter codes to 1-letter codes:
res3_to_res1 = {
    "ASP": "D",
    "GLU": "E",
    "ARG": "R",
    "HIS": "H",
    "LYS": "K",
    "CYS": "C",
    "TYR": "Y",
    "PL9": "MQ8",
}


class Conformer:
    """Minimal Conformer class for use in microstate analysis.
    Attributes: iconf, confid, ires, resid, crg.
    """

    def __init__(self):
        self.iconf = 0
        self.confid = ""
        self.ires = 0
        self.resid = ""
        self.crg = 0.0

    def __str__(self):
        return (
            f"iconf={self.iconf}, confid={self.confid}, " f"ires={self.ires}, resid={self.resid}, crg={self.crg}"
        )

    def load_from_head3lst(self, line):
        fields = line.split()
        self.iconf = int(fields[0]) - 1
        self.confid = fields[1]
        self.resid = self.confid[:3] + self.confid[5:11]
        self.crg = float(fields[4])


def read_conformers(head3_path: str = "head3.lst") -> list:
    """Load conformerrs from given head3.lst path;
    Uses ./head3.lst by default; returns empty list if file not found.
    """
    if not Path(head3_path).exists():
        return []

    conformers = []
    with open(head3_path) as h3:
        lines = h3.readlines()[1:]

    for line in lines:
        conf = Conformer()
        conf.load_from_head3lst(line)
        conformers.append(conf)

    return conformers


def reader_gen(fpath: Path):
    """
    Generator function yielding a file line.
    """
    with open(fpath) as fh:
        for line in fh:
            yield line


class Microstate:
    """Sortable class for mcce microstates."""

    def __init__(self, state: list, E: float, count: int):
        self.state = state
        self.E = E
        self.count = count

    def __str__(self):
        return f"Microstate(\n\tcount={self.count:,},\n\tE={self.E:,},\n\tstate={self.state}\n)"

    def _check_operand(self, other):
        """Fails on missing attribute."""
        if not (hasattr(other, "state") and hasattr(other, "E") and hasattr(other, "count")):
            return NotImplemented("Comparison with non Microstate object.")
        return

    def __eq__(self, other):
        self._check_operand(other)
        return (self.state, self.E, self.count) == (
            other.state,
            other.E,
            other.count,
        )

    def __lt__(self, other):
        self._check_operand(other)
        return (self.state, self.E, self.count) < (
            other.state,
            other.E,
            other.count,
        )


class MSout:
    __slots__ = [
        "T",
        "pH",
        "Eh",
        "N_ms",
        "N_uniq",
        "lowest_E",
        "highest_E",
        "average_E",
        "fixed_iconfs",
        "free_residues",
        "iconf2ires",
        "microstates",
    ]

    def __init__(self, fname):
        self.T = 298.15
        self.pH = 7.0
        self.Eh = 0.0
        self.N_ms = 0
        self.N_uniq = 0
        self.lowest_E = 0.0
        self.highest_E = 0.0
        self.average_E = 0.0
        self.fixed_iconfs = []
        self.free_residues = []  # free residues, referred by conformer indices, iconf
        self.iconf2ires = {}  # from conformer index to free residue index
        self.microstates = {}  # dict of Microstate objects

        self.load_msout(fname)

    def load_msout(self, fname):
        """Process the 'msout file' to populate these attributes:
        - T, pH, Eh (floats)
        - fixed_iconfs (list)
        - free_residues (list)
        - iconf2ires (dict)
        - N_ms, N_uniq (int)
        - microstates (dict)
        - lowest_E, average_E, highest_E (float)
        """
        found_mc = False
        newmc = False
        msout_data = reader_gen(fname)
        for i, line in enumerate(msout_data, start=1):
            line = line.strip()
            if not line or line[0] == "#":
                continue

            if i == 1:
                fields = line.split(",")
                for field in fields:
                    key, value = field.split(":")
                    key = key.strip().upper()
                    value = float(value)
                    if key == "T":
                        self.T = value
                    elif key == "PH":
                        self.pH = value
                    elif key == "EH":
                        self.Eh = value
            if i == 2:
                key, value = line.split(":")
                if key.strip() != "METHOD" or value.strip() != "MONTERUNS":
                    logger.critical("This file %s is not a valid microstate file" % fname)
                    sys.exit(-1)
            if i == 4:
                _, iconfs = line.split(":")
                self.fixed_iconfs = [int(i) for i in iconfs.split()]

            if i == 6:  # free residues
                _, residues_str = line.split(":")
                residues = residues_str.split(";")
                self.free_residues = []
                for f in residues:
                    if f.strip():
                        self.free_residues.append([int(i) for i in f.split()])
                for i_res in range(len(self.free_residues)):
                    for iconf in self.free_residues[i_res]:
                        self.iconf2ires[iconf] = i_res
            else:
                # find the next MC record
                if line.startswith("MC:"):
                    found_mc = True
                    newmc = True
                    continue

                if newmc:
                    f1, f2 = line.split(":")
                    current_state = [int(c) for c in f2.split()]
                    newmc = False
                    continue

                if found_mc:
                    fields = line.split(",")
                    if len(fields) >= 3:
                        state_e = float(fields[0])
                        count = int(fields[1])
                        flipped = [int(c) for c in fields[2].split()]
                        for ic in flipped:
                            ir = self.iconf2ires[ic]
                            current_state[ir] = ic

                        ms = Microstate(list(current_state), state_e, count)
                        key = ",".join(["%d" % i for i in ms.state])
                        if key in self.microstates:
                            self.microstates[key].count += ms.count
                        else:
                            self.microstates[key] = ms

        # find N_ms, lowest, highest, averge E
        E_sum = 0.0
        msvals = self.microstates.values()
        self.N_uniq = len(msvals)
        ms = next(iter(msvals))
        lowest_E = ms.E
        highest_E = ms.E
        for ms in msvals:
            self.N_ms += ms.count
            E_sum += ms.E * ms.count
            if ms.E < lowest_E:
                lowest_E = ms.E
            elif ms.E > highest_E:
                highest_E = ms.E

        self.lowest_E = lowest_E
        self.average_E = E_sum / self.N_ms
        self.highest_E = highest_E

        return

    def get_fixed_res_crg_dict(self, conformers: list) -> dict:
        """Map fixed residues 'conformer.resid' to their charges.
        Return a dict with key=resid, value=charge.
        """
        fixed_res_crg_dict = defaultdict(float)
        for conf in conformers:
            if conf.iconf in self.fixed_iconfs:
                fixed_res_crg_dict[conf.resid] = conf.crg

        return dict(fixed_res_crg_dict)

    def get_fixed_res_crg(self, data: Union[list, dict]) -> float:
        """Return the charge contributed by fixed conformers.
        If data is a list, assumed: data=conformers, else
        if data is a dict, assumed format: dict[conf.resid] = conf.crg
        as in .get_fixed_res_crg_dict.
        """
        if isinstance(data, list):
            return sum([conf.crg for conf in data if conf.iconf in self.fixed_iconfs])
        return sum(data.values())

    # TODO: Keep?
    def get_preset_energy_bounds(self) -> dict:
        """Return a dict to store typical energy bounds that can be used to process
        charge microstates; e.g. save the unique charge microstates that are within these
        bounds to list & plot their correlation.
        Key: An integer (order of computation), starting at 1;
        Value: A 2-tuple: (energy1, energy2).
         1 :: (None, None) => all cms
         2 :: cms within 1 KT unit (1.36 kcal/mol) of lowest microstate energy
         3 :: cms within +/- 0.5 pH unit (+/- 0.68 kcal/mol) of average microstate energy
         4 :: cms within 1 KT unit (1.36 kcal/mol) of highest microstate energy
        """
        ebounds = {
            1: (None, None),
            2: (self.lowest_E, self.lowest_E + ph2Kcal),
            3: (self.average_E - half_ph, self.average_E + half_ph),
            4: (self.highest_E - ph2Kcal, self.highest_E),
        }
        return ebounds

    def get_sampled_ms(
        self,
        size: int,
        kind: str = "deterministic",
        seed: Union[None, int] = None,
    ) -> list:
        """
        Implement a sampling of MSout.microstates depending on `kind`.
        Args:
            size (int): sample size
            kind (str, 'deterministic'): Sampling kind: one of ['deterministic', 'random'].
                 If 'deterministic', the microstates in ms_list are sampled at regular intervals
                 otherwise, the sampling is random. Case insensitive.
            seed (int, None): For testing purposes, fixes random sampling.
        Returns:
            A list of lists: [[selection index, selected microstate], ...]
        """
        if not len(self.microstates):
            logger.error("The microstates dict is empty.")
            return []

        kind = kind.lower()
        if kind not in ["deterministic", "random"]:
            raise ValueError(f"Values for `kind` are 'deterministic' or 'random'; Given: {kind}")

        ms_sampled = []
        ms_list = list(self.microstates.values())
        counts = ms_counts(ms_list)  # total number of ms
        sampled_cumsum = np.cumsum([mc.count for mc in ms_list])

        if kind == "deterministic":
            sampled_ms_indices = np.arange(size, counts - size, counts / size, dtype=int)
        else:
            rng = np.random.default_rng(seed=seed)
            sampled_ms_indices = rng.integers(low=0, high=counts, size=size, endpoint=True)

        for i, c in enumerate(sampled_ms_indices):
            ms_sel_index = np.where((sampled_cumsum - c) > 0)[0][0]
            ms_sampled.append([ms_sel_index, ms_list[ms_sel_index]])

        return ms_sampled

    def sort_microstates(self, sort_by: str = "E", sort_reverse: bool = False) -> list:
        """Return the list of microstates sorted by one of these attributes: ["count", "E"],
        and in reverse order (descending) if sort_reverse is True.
        Args:
        microstates (list): list of Microstate instances;
        sort_by (str, "E"): Attribute as sort key;
        sort_reverse (bool, False): Sort order: ascending if False (default), else descending.
        Return None if 'sort_by' is not recognized.
        """

        if sort_by not in ["count", "E"]:
            logger.error("Not sorted: 'sort_by' must be a valid microstate attribute; choices: ['count', 'E']")
            return list(self.microstates.values())

        return sorted(
            list(self.microstates.values()),
            key=operator.attrgetter(sort_by),
            reverse=sort_reverse,
        )

    def __str__(self):
        return (
            f"\nNumber of microstates: {self.N_ms:,}\nNumber of unique microstates: "
            f"{self.N_uniq:,}\nEnergies: lowest_E: {self.lowest_E:,.2f}; average_E: "
            f"{self.average_E:,.2f}; highest_E: {self.highest_E:,.2f}\n"
            f"Free residues: {len(self.free_residues):,}\n"
            f"Fixed residues: {len(self.fixed_iconfs):,}\n"
        )


class WeightedCorr:
    def __init__(
        self,
        xyw: pd.DataFrame = None,
        x: pd.Series = None,
        y: pd.Series = None,
        w: pd.Series = None,
        df: pd.DataFrame = None,
        wcol: str = None,
        cutoff: float = 0.02,
    ):
        """Weighted Correlation class.
        To instantiate WeightedCorr, either supply:
          1. xyw as pd.DataFrame,
          2. 3 pd.Series: (x, y, w),
          3. a pd.DataFrame and the name of the weight column.
        Args:
          xyw: pd.DataFrame with shape(n,3) containing x, y, and w columns;
          x: pd.Series (n,) containing values for x;
          y: pd.Series (n,) containing values for y;
          w: pd.Series (n,) containing weights;
          df: pd.Dataframe (n,m+1) containing m phenotypes and a weight column;
          wcol: str column of the weight in the df argument.
          cutoff: if given, return values whose absolute values are greater.
        Usage:
          ```
          # input as DataFrame and weight column name:
          wcorr = WeightedCorr(df=input_df, wcol="Count", cutoff=0.01)(method='pearson')

          # input as a DataFrame subset:
          wcorr = WeightedCorr(xyw=df[["xcol", "ycol", "wcol"]])(method='pearson')
          ```
        """
        self.cutoff = cutoff

        if (df is None) and (wcol is None):
            if np.all([i is None for i in [xyw, x, y, w]]):
                raise ValueError("No data supplied")

            if not ((isinstance(xyw, pd.DataFrame)) != (np.all([isinstance(i, pd.Series) for i in [x, y, w]]))):
                raise TypeError("xyw should be a pd.DataFrame, or x, y, w should be pd.Series")

            xyw = pd.concat([x, y, w], axis=1).dropna() if xyw is None else xyw.dropna()
            self.x, self.y, self.w = (pd.to_numeric(xyw[i], errors="coerce").values for i in xyw.columns)
            self.df = None

        elif (wcol is not None) and (df is not None):
            if (not isinstance(df, pd.DataFrame)) or (not isinstance(wcol, str)):
                raise ValueError("df should be a pd.DataFrame and wcol should be a string")

            if wcol not in df.columns:
                raise KeyError("wcol not found in column names of df")

            if not df.shape[0] > 1:
                sys.exit("Too few rows for correlation.")

            cols = df.columns.to_list()
            _ = cols.pop(cols.index(wcol))
            self.df = df.loc[:, cols]
            self.w = pd.to_numeric(df.loc[:, wcol], errors="coerce")

        else:
            raise ValueError("Incorrect arguments specified, please specify xyw, or (x, y, w) or (df, wcol)")

    def _wcov(self, x, y, ms):
        return np.sum(self.w * (x - ms[0]) * (y - ms[1]))

    def _pearson(self, x=None, y=None) -> float:
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        mx, my = (np.sum(i * self.w) / np.sum(self.w) for i in [x, y])

        # needed for unchanging values (fixed res):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                wcov = self._wcov(x, y, [mx, my]) / np.sqrt(
                    self._wcov(x, x, [mx, mx]) * self._wcov(y, y, [my, my])
                )
            except RuntimeWarning:
                wcov = 0

        if abs(wcov) > self.cutoff:
            return wcov
        else:
            return 0

    def _wrank(self, x):
        (unique, arr_inv, counts) = np.unique(rankdata(x), return_counts=True, return_inverse=True)
        a = np.bincount(arr_inv, self.w)
        return (np.cumsum(a) - a)[arr_inv] + ((counts + 1) / 2 * (a / counts))[arr_inv]

    def _spearman(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        return self._pearson(self._wrank(x), self._wrank(y))

    def __call__(self, method: str = "pearson") -> Union[float, pd.DataFrame]:
        """
        WeightedCorr call method.
        Args:
          method (str, "pearson"): Correlation method to be used:
                                   'pearson' for pearson r, 'spearman' for spearman rank-order correlation.
        Return:
          - The correlation value as float if xyw, or (x, y, w) were passed to __init__;
          - A m x m pandas.DataFrame holding the correlaton matrix if (df, wcol) were passed to __init__.
        """
        method = method.lower()
        if method not in CORR_METHODS:
            raise ValueError(f"`method` should be one of {CORR_METHODS}.")

        # define which of the defined methods to use:
        cor = {"pearson": self._pearson, "spearman": self._spearman}[method]

        if self.df is None:  # run the method over series
            return cor()
        else:
            # run the method over matrix
            df_out = pd.DataFrame(np.nan, index=self.df.columns, columns=self.df.columns)
            for i, x in enumerate(self.df.columns):
                for j, y in enumerate(self.df.columns):
                    if i >= j:
                        df_out.loc[x, y] = cor(
                            x=pd.to_numeric(self.df[x], errors="coerce"),
                            y=pd.to_numeric(self.df[y], errors="coerce"),
                        )
                        df_out.loc[y, x] = df_out.loc[x, y]

            if self.cutoff is not None:
                # values will be all 0 when cutoff is applied in cor():
                msk = df_out == 0
                df_out = df_out.loc[~msk.all(axis=1), ~msk.all(axis=0)]

            return df_out


# For splitting a string with re. Remove punctuation and spaces:
re_split_pattern = re.compile(r"[\s{}]+".format(re.escape(string.punctuation)))
# for splitting lists of residues for correlation: keep underscore
correl_split_pattern = re.compile(r"[\s{}]+".format(re.escape(string.punctuation.replace("_", ""))))


def split_spunct(text, upper=True, pattern: re.Pattern = re_split_pattern) -> list:
    """Split text on space and punctuation."""
    if not text:
        return []
    if upper:
        text = text.upper()
    return re.split(pattern, text)


def sort_resoi_list(resoi_list: list) -> list:
    """Return the input 'res of interest' list with ionizable residues in
    the same order as msa.IONIZABLES, i.e.:
    acid, base, polar, N term, C term, followed by user provided res, sorted.
    """
    if not resoi_list:
        return []

    userlst = [res.upper() for res in resoi_list]
    ioniz = deepcopy(IONIZABLES)

    ioniz_set = set(ioniz)
    sym_diff = ioniz_set.symmetric_difference(userlst)
    new_res = sym_diff.difference(ioniz_set)
    removal = sym_diff.difference(new_res)
    if removal:
        for res in removal:
            ioniz.pop(ioniz.index(res))

    return ioniz + sorted(new_res)


def params_main(ph: float = 7) -> dict:
    """Obtain cms_analysis main parameters dict with default values for given ph."""
    params_defaults = {
        # Most values are strings to match the values in the dicts returned by `load_param_file`.
        "mcce_dir": ".",
        "output_dir": f"crgms_corr_ph{ph}",
        "list_head3_ionizables": "False",
        #Do not output file 'all_crg_count_res_ph7.csv'
        #"main_csv": "all_crg_count_res_ph7.csv",
        "all_res_crg_csv": "all_res_crg_status.csv",
        "res_of_interest_data_csv": "crg_count_res_of_interest.csv",
        "msout_file": f"pH{ph}eH0ms.txt",
        "residue_kinds": IONIZABLES,
        "correl_resids": None,
        "corr_method": "pearson",
        "corr_cutoff": "0.02",
        "n_clusters": "5",
        "fig_show": "False",
        "energy_histogram.save_name": f"enthalpy_dist.png",
        "energy_histogram.fig_size": "(8,8)",
        "corr_heatmap.save_name": f"corr.png",
        "corr_heatmap.fig_size": "(20, 8)",
    }

    return params_defaults


def params_histograms() -> dict:
    """Obtain cms_analysis histogram parameters dict with default values for given ph."""
    params_defaults = {
        "charge_histogram0": {
            "bounds": "(None, None)",
            "title": "Charge Microstates Energy",
            "save_name": "crgms_logcount_vs_E.png",
        },
        "charge_histogram1": {
            "bounds": "(Emin, Emin + 1.36)",
            "title": "ChargeMicrostates Energy within 1.36 kcal/mol of Lowest",
            "save_name": "crgms_logcount_vs_lowestE.png",
        },
        "charge_histogram2": {
            "bounds": "(Eaver - 0.68, Eaver + 0.68)",
            "title": "Charge Microstates Energy within 0.5 pH (0.68 kcal/mol) of Mean",
            "save_name": "crgms_logcount_vs_averE.png",
        },
        "charge_histogram3": {
            "bounds": "(Emax - 1.36, Emax)",
            "title": "Charge Microstates Energy within 1.36 kcal/mol of Highest",
            "save_name": "crgms_logcount_vs_highestE.png",
        },
    }

    return params_defaults


def list_head3_ionizables(h3_fp: Path, as_string: bool=True) -> list:
    """Return the list of ionizable resids from head3.lst.
    May be used to choose 'correl_resids'.
    """
    h3_fp = Path(h3_fp)
    if h3_fp.name != "head3.lst":
        print(f"File name not 'head3.lst': {h3_fp!s}")
        return []
    if not h3_fp.exists():
        print(f"Not found: {h3_fp!s}")
        return []
    
    h3_ioniz_res = []
    h3_df = pd.read_csv(h3_fp, sep="\s+")
    h3_ioniz_res = list(dict((f"{res[:3]}{res[5:11]}", "") for res in h3_df.iConf if res[:3] in IONIZABLES).keys())
    if not as_string:
        return h3_ioniz_res

    res_lst = "[\n"
    for res in h3_ioniz_res:
        res_lst += f"{res!r},\n"
    res_lst += "]"
    
    return res_lst


def load_crgms_param(filepath: str) -> Tuple[dict, dict]:
    """Load parameters file into two dicts; the second one is used for processing
    the charge histograms calculations & plotting.
    """
    fp = Path(filepath)
    if not fp.exists():
        return FileNotFoundError(fp)

    crgms_dict = {}
    correl_lines = []

    with open(filepath) as f:
        # data lines:
        lines = [line.strip() for line in f.readlines()
                 if line.strip()
                 and not line.strip().startswith("#")
                 ]

    multi_found = False
    for line in lines:
        if not multi_found:
            try:
                key, rawval = line.split("=")
            except ValueError:
                raise ValueError("Malformed entry: single equal sign required.")

            key, rawval = key.strip(), rawval.strip()
            if rawval.startswith("("):
                # check tuple:
                if rawval.endswith(")"):
                    val = rawval
                else:
                    raise ValueError("Malformed tuple: must be (x,y) on same line.")
            elif rawval.startswith("["):
                # check list on same line:
                if rawval.endswith("]"):
                    if key == "residue_kinds":
                        val = sort_resoi_list([v for v in split_spunct(rawval[1:-1].strip()) if v])
                    else:
                        # correl_resids on same line
                        val = [v for v in split_spunct(rawval[1:-1].strip(), pattern=correl_split_pattern) if v]
                else:
                    if key == "residue_kinds":
                        sys.exit(("Malformed residue_kinds entry, ']' not found: "
                                  "list within square brackets must be on the same line.")
                                 )
                    elif key == "correl_resids":
                        multi_found = True
                        continue
            else:
                # all others: strings:
                val = rawval
            if not multi_found:
                crgms_dict[key] = val
        else:
            rawval = line.strip()
            if not rawval.endswith("]"):
                correl_lines.extend([v for v in split_spunct(rawval, pattern=correl_split_pattern) if v])
                continue
            if rawval.endswith("]"):
                multi_found = False
                continue

    if correl_lines:
        crgms_dict["correl_resids"] = correl_lines

    p, e = crgms_dict.get("msout_file", "pH7eH0ms.txt")[:-4].lower().split("eh")
    ph = p.removeprefix("ph")
    crgms_dict["ph"] = ph
    crgms_dict["eh"] = e.removesuffix("ms")

    charge_histograms = defaultdict(dict)
    remove_keys = []
    for k in crgms_dict:
        if k.startswith("charge_histogram"):
            v = crgms_dict[k]
            k1, k2 = k.split(".")
            charge_histograms[k1].update({k2: v})
            remove_keys.append(k)

    for k in remove_keys:
        crgms_dict.pop(k)

    # Add default params:
    main_params = params_main(ph=ph)
    for k in main_params:
        if crgms_dict.get(k) is None:
            crgms_dict[k] = main_params[k]

    # Add params for unbounded histogram if none were given:
    if not charge_histograms:
        charge_histograms["charge_histogram0"] = params_histograms()["charge_histogram0"]

    return crgms_dict, dict(charge_histograms)


def ms_counts(microstates: Union[dict, list]) -> int:
    """Calculate total counts of microstates, which can be given
    as a list or a dict.
    """
    if isinstance(microstates, dict):
        return sum(ms.count for ms in microstates.values())
    else:
        return sum(ms.count for ms in microstates)


def get_ms_charge(ms: Microstate, conformers: list) -> Union[float, None]:
    """Compute a microstate's net charge."""
    if conformers:
        return sum(conformers[ic].crg for ic in ms.state)
    else:
        logger.error("Module conformers list is empty.")
        return None


def iconf2crg(conformers: list) -> dict:
    """Map mcce conformers indices to their charges."""
    return {conf.iconf: conf.crg for conf in conformers}


def iconf2ires(free_residues: list) -> dict:
    """Map conf indices to parent res index.
    Note: Reverse mapping of MSout.free_residue.
    """
    d = {}
    for i_res in range(len(free_residues)):
        for iconf in free_residues[i_res]:
            d[iconf] = i_res
    return d


def free_res2sumcrg_df(microstates: Union[dict, list], free_res: list, conformers: list) -> pd.DataFrame:
    """
    Given a list of microstates and free residues, convert to net charge of each free residue
    into a pandas.DataFrame.
    Returns:
      None if conformers list is empty.
    """
    if not conformers:
        logger.error("The conformers list is empty!")
        return None

    if isinstance(microstates, dict):
        microstates = list(microstates.values())

    charges_total = defaultdict(float)
    N_ms = 0
    for ms in microstates:
        N_ms += ms.count
        for ic in ms.state:
            charges_total[conformers[ic].resid] += conformers[ic].crg * ms.count

    for x in charges_total:
        tot = charges_total[x]
        charges_total[x] = round(tot / N_ms, 0)

    return pd.DataFrame(charges_total.items(), columns=["Residue", "crg"])


def free_residues_to_df(free_res: list, conformers: list, colname: str = "FreeRes") -> pd.DataFrame:
    """Return the free residues' ids in a pandas DataFrame."""
    return pd.DataFrame([conformers[res[0]].resid for res in free_res], columns=[colname])


def fixed_residues_info(
    fixed_iconfs: list, conformers: list, res_of_interest: list = IONIZABLES
) -> Tuple[float, pd.DataFrame, dict]:
    """
    Return a 3-tuple: 0: background charge from all fixed residues;
                      1: fixed res crg df for res listed in res_of_interest, if any;
                      2: dict: maps fixed resid to crg (source for item 1).
    Note:
      res_of_interest defaults to the list of ionizable residues.
    """
    dd = defaultdict(float)
    for conf in conformers:
        if conf.iconf in fixed_iconfs:
            dd[conf.resid] = int(conf.crg)
    fixed_backgrd_charge = sum(dd.values())
    if res_of_interest:
        dd = {k: dd[k] for k in dd if k[:3] in res_of_interest}
    fixed_res_crg_df = pd.DataFrame(dd.items(), columns=["Residue", "crg"])

    return fixed_backgrd_charge, fixed_res_crg_df, dd


def ms2crgms(ll: list, dd: dict) -> list:
    """
    Given a list with format: [[ms.E, ms.count, ms.state], ] sorted by E,
    and a dict mapping conformer indices to their charges, convert the conformer
    state list to to conformer charges.
    """
    crg_lst = [
        [
            y[0],
            y[1],
            [ms2crgms(x, dd) if isinstance(x, list) else dd.get(x, x) for x in y[2]],
        ]
        for y in ll
    ]

    return crg_lst


def filter_fixed_from_choose_res(fixed_resoi_df: pd.DataFrame, choose_res: list) -> list:
    fixed_res = fixed_resoi_df.Residue.tolist()
    return [res for res in choose_res if res not in fixed_res]


def rename_order_residues(crgms_data: pd.DataFrame, chosen_free_res: list):
    """Rename resid in df columns if in chosen_free_res."""
    rename_dict = {}
    acid_list = []
    base_list = []
    polar_list = []
    ub_q_list = []
    non_res_list = []

    # exclude Occupancy column expected as last column:
    # NTRA0001_	GLUA0007_	GLUA0035_	ASPA0048_	ASPA0052_	TYRA0053_	Count	Occupancy
    for col in crgms_data.columns[:-1]:
        if col in chosen_free_res:
            residue_number = col[4:8]

            if residue_number.isdigit():  # Check if the substring is numeric
                # remove leading zeros
                rename_dict[col] = col[3] + "_" + col[:3] + str(int(residue_number))
            else:
                rename_dict[col] = col[3] + "_" + col[:3] + residue_number

    rename_dict["Count"] = "Count"

    for k in rename_dict:
        y = rename_dict[k]
        if y == "Count":
            non_res_list.append(y)
            continue

        resid = y[2:5]
        if resid == "PL9":
            ub_q_list.append(res3_to_res1.get(resid, resid) + y[5:])
            continue

        # resid won't be shortened if not in msa.res3_to_res1:
        resout = y[:1] + res3_to_res1.get(resid, resid) + y[5:]
        # update dict:
        rename_dict[k] = resout
        if resid in ACIDIC_RES:
            acid_list.append(resout)
        elif resid in BASIC_RES:
            base_list.append(resout)
        elif resid in POLAR_RES:
            polar_list.append(resout)
        else:
            # unaccounted for:
            non_res_list.append(resout)

    col_order_list = acid_list + polar_list + base_list + ub_q_list + non_res_list
    file_out = crgms_data.rename(rename_dict, axis=1)
    file_out = file_out[col_order_list]

    return file_out


def choose_res_data(df: pd.DataFrame, choose_res: list) -> pd.DataFrame:
    """Group the df by the given list.
    Returns:
     A pandas.DataFrame grouped by choose_res and sorted by 'Count',
    Note:
     - df must have a 'Count' column.
    """
    if not choose_res:
        raise TypeError("Error: empty list given for df.groupby 'by' argument.")

    df_choose_res = df.groupby(choose_res).Count.sum().reset_index()
    df_res_sort = df_choose_res.sort_values(by="Count", ascending=False).reset_index(drop=True)

    return df_res_sort


def find_uniq_crgms_count_order(
    crg_list_ms: list, begin_energy: float = None, end_energy: float = None
) -> Tuple[list, list, list, list]:
    """
    Args:
      crg_list_ms (list): List of charge microstate, assumed sorted.
      begin_energy (float, None): Lower energy bound.
      end_energy (float, None): Upper energy bound.

    Returns:
      A 4-tuple of lists:
        0: all_crg_ms_unique,
        1: all_count,
        2: unique_crg_state_order,
        3: energy_diff_all (used in plot)

    Notes:
        1. To filter crg_list_ms based on energy, both energy bounds are needed.
        2. Variable unique_crg_state_order gives the order of unique charge state based on energy.
           Lowest energy charge state will give the order 1; the second will give the order 2.
    """
    if not begin_energy and not end_energy:
        logger.info("No energy bounds given: All microstates are selected.")
        begin_energy = crg_list_ms[0][0]
        end_energy = crg_list_ms[-1][0]

        logger.info(f"Number of charge ms: {len(crg_list_ms):,}")

    elif begin_energy and end_energy:
        # both energy bounds given; filter the input list energy accordingly:
        crg_list_ms = [[x[0], x[1], x[2]] for x in crg_list_ms if x[0] >= begin_energy and x[0] <= end_energy]
        logger.info(f"Number of energy-filtered charge ms: {len(crg_list_ms):,}")
    else:
        sys.exit("Both energy bounds are needed.")

    crg_all_count = {}
    unique_crg_state_order = 1
    for E, count, state in crg_list_ms:
        if tuple(state) not in crg_all_count:
            # initial k,v:
            # k = tuple(crgms state); v = [count, [energy], [unique_crg_state_order]]
            crg_all_count[tuple(state)] = [
                count,
                [E],
                unique_crg_state_order,
            ]
            unique_crg_state_order += 1
        else:
            # same state: increment the total count of ms
            crg_all_count[tuple(state)][0] += count

            # get the maximum and minimum energy
            min_energy = min(min(crg_all_count[tuple(state)][1]), E)
            max_energy = max(max(crg_all_count[tuple(state)][1]), E)

            # rest the energy list with the min and max energy so far:
            crg_all_count[tuple(state)][1].clear()
            crg_all_count[tuple(state)][1].append(min_energy)
            crg_all_count[tuple(state)][1].append(max_energy)

    # make a list of crgms counts, unique charge microstate, order and E difference.
    all_crg_ms_unique = []
    all_count = []
    energy_diff_all = []
    unique_crg_state_order = []

    for k in crg_all_count:
        all_crg_ms_unique.append(list(k))

        all_count.append(crg_all_count[k][0])
        unique_crg_state_order.append(crg_all_count[k][2])
        # E list: holds single value for unique crg state, or [min, max]
        E = crg_all_count[k][1]
        if len(E) == 2:
            energy_diff_all.append(round(E[1] - E[0], 3))
        else:
            energy_diff_all.append(0.0)

    logger.info(f"Number of unique charge ms: {len(all_crg_ms_unique):,}")

    return all_crg_ms_unique, all_count, unique_crg_state_order, energy_diff_all


def combine_all_free_fixed_residues(free_res_crg_df: pd.DataFrame, fixed_res_crg_df: pd.DataFrame) -> pd.DataFrame:
    free_res_crg_df["status"] = "free"
    fixed_res_crg_df["status"] = "fixed"
    df = pd.concat([free_res_crg_df, fixed_res_crg_df])
    df.set_index("Residue", inplace=True)

    return df.T


# TODO: Deprecate?
def combine_free_fixed_residues0(
    free_res_crg_count_df: pd.DataFrame, fixed_residue_df: pd.DataFrame
) -> pd.DataFrame:

    df_fixed_res_crg = fixed_residue_df.T
    df_fixed_res_crg.columns = df_fixed_res_crg.iloc[0]
    df_fixed_res_crg = df_fixed_res_crg.iloc[1:, :].reset_index(drop=True)
    df_fixed_res_dup = pd.concat([df_fixed_res_crg] * len(free_res_crg_count_df), ignore_index=True)
    df_all = free_res_crg_count_df.join(df_fixed_res_dup)

    return df_all


def concat_crgms_dfs(
    unique_crgms: list,
    ms_count: list,
    ms_order: list,
    free_res_df: pd.DataFrame,
    background_charge: float,
    res_of_interest: list = IONIZABLES,
) -> pd.DataFrame:

    crgms_count_df = pd.concat(
        [
            pd.DataFrame(unique_crgms).T,
            pd.DataFrame(ms_count, columns=["Count"]).T,
            pd.DataFrame(ms_order, columns=["Order"]).T,
        ]
    )
    all_crg_count_res = pd.concat([free_res_df, crgms_count_df], axis=1)
    all_crg_count_res.loc["Count", "Residue"] = "Count"
    all_crg_count_res.loc["Order", "Residue"] = "Order"
    all_crg_count_res.set_index("Residue", inplace=True)

    # sort by Count descendingly:
    all_crg_count_res = all_crg_count_res.sort_values(by="Count", axis=1, ascending=False)
    all_crg_count_res.columns = range(all_crg_count_res.shape[1])

    # transpose and change index to Order; add occ & sumcrg:
    all_crg_count_res = all_crg_count_res.T.set_index("Order")
    all_crg_count_res["Occupancy"] = round(all_crg_count_res["Count"] / sum(all_crg_count_res["Count"]), 3)
    all_crg_count_res["Sum_crg_protein"] = all_crg_count_res.iloc[:, :-2].sum(axis=1) + background_charge

    crg_count_res = all_crg_count_res.copy()
    for c in all_crg_count_res.columns:
        if c[:3] not in res_of_interest and c not in ["Occupancy", "Count", "Sum_crg_protein"]:
            crg_count_res.drop([c], axis=1, inplace=True)

    return crg_count_res


def add_fixed_res_crg(main_df: pd.DataFrame, fixed_df: pd.DataFrame) -> pd.DataFrame:
    """Return a df which is main_df augmented with data from fixed_df;
    The totals columns order is preserved & the index ("Order") is sorted.
    """
    out = main_df.copy()
    N = out.shape[0]
    tot_cols = out.columns[-3:].tolist()

    new_df = pd.DataFrame({v[0]: [v[1]] * N for v in fixed_df.values})
    for c in new_df.columns:
        out[c] = new_df[c].values
    try:
        out.sort_values(by="Count", ascending=False, inplace=True)
    except (IndexError, KeyError):
        out.sort_index(inplace=True)
    res_cols = out.columns.tolist()
    for tc in tot_cols:
        res_cols.remove(tc)

    return out[res_cols + tot_cols]


def changing_residues_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return transposed df minus unchanging residues as per stdev."""

    # exclude Occupancy & Sum_crg_protein columns and transpose:
    new_df = df.iloc[:, :-2].T
    sorted_cols = sorted(list(new_df.columns))
    new_df = new_df[sorted_cols]
    new_df["std"] = new_df.std(axis=1).round(3)
    msk = new_df["std"] != 0
    change_df = new_df.loc[msk].T[:-1].reset_index(drop=True)
    # subtract 'Count' column:
    logger.info(f"Number of residues that change protonation state: {len(change_df.columns)-1}")

    return change_df


# >>> plotting functions ....................................
def unique_crgms_histogram(
    charge_ms_info: tuple,
    background_charge: float,
    fig_title: str,
    out_dir: Path,
    save_name: str,
    add_hue: bool = False,
    show: bool = False,
):
    """
    Visualize which tautomer charge state is most populated.
    This includes the background charge.
    To show the energy ranges in the plot hue parameter, set add_hue to True.
    """
    x_av = [sum(x) + background_charge for x in charge_ms_info[0]]
    y_av = [math.log10(x) for x in charge_ms_info[1]]

    hue_data, palet = None, None
    if add_hue:
        hue_data = charge_ms_info[3]  # E differences
        palet = "viridis"

    palet
    g1 = sns.JointGrid(marginal_ticks=True, height=6)
    ax = sns.scatterplot(
        x=x_av,
        y=y_av,
        hue=hue_data,
        palette=palet,
        size=charge_ms_info[1],
        sizes=(10, 200),
        legend="brief",
        ax=g1.ax_joint,
    )

    fs = 14  # font size for axes labels and title
    ax.set_xticks(range(int(min(x_av)), int(max(x_av)) + 1))
    ax.set_xlabel("Charge", fontsize=fs)
    ax.set_ylabel("log$_{10}$(Count)", fontsize=fs)
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc=2, borderaxespad=0.0)

    ax2 = sns.histplot(x=x_av, linewidth=2, discrete=True, ax=g1.ax_marg_x)
    ax2.set_ylabel(None, fontsize=fs)
    g1.ax_marg_y.set_axis_off()
    g1.fig.subplots_adjust(top=0.9)
    if fig_title:
        g1.fig.suptitle(fig_title, fontsize=fs)

    fig_fp = out_dir.joinpath(save_name)
    g1.savefig(fig_fp, dpi=300, bbox_inches="tight")
    logger.info(f"Figure saved: {fig_fp}")

    if show:
        plt.show()
    else:
        plt.close()

    return


def ms_energy_histogram(
    ms_by_enrg: list, out_dir: Path, save_name: str = "enthalpy_dist.png", show: bool = False, fig_size=(8, 8)
):
    """
    Plot the histogram of microstates energy.
    Args:
      ms_by_enrg: List of microstates sorted by energy.

    """
    energy_lst_count = np.asarray(
        [a for a, f in zip([x[0] for x in ms_by_enrg], [x[1] for x in ms_by_enrg]) for _ in range(f)]
    )
    skewness, mean, std = skewnorm.fit(energy_lst_count)

    fig = plt.figure(figsize=fig_size)
    fs = 14  # fontsize

    graph_hist = plt.hist(energy_lst_count, bins=100, alpha=0.6)
    Y = graph_hist[0]
    y = skewnorm.pdf(energy_lst_count, skewness, mean, std)

    pdf_data = Y.max() / max(y) * y
    plt.plot(energy_lst_count, pdf_data, label="approximated skewnorm", color="k")
    plt.title(f"{skewness = :.2f} {mean = :.2f} {std = :.2f}", fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.xlabel("Microstate Energy (Kcal/mol)", fontsize=fs)
    plt.ylabel("Count", fontsize=fs)
    plt.tick_params(axis="x", direction="out", length=8, width=2)
    plt.tick_params(axis="y", direction="out", length=8, width=2)

    fig_fp = out_dir.joinpath(save_name)
    fig.savefig(fig_fp, dpi=300, bbox_inches="tight")
    logger.info(f"Histogram figure saved as {fig_fp}")

    if show:
        plt.show()
    else:
        plt.close()

    return


HEATMAP_SIZE = (20, 8)


def corr_heatmap(
    df_corr: pd.DataFrame,
    out_dir: Path = None,
    save_name: str = "corr.png",
    check_allzeros: bool = True,
    show: bool = False,
    lower_tri=False,
    fig_size: Tuple[float, float] = HEATMAP_SIZE,
):
    """Produce a heatmap from a correlation matrix.
    Args:
     - df_corr (pd.DataFrame): Correlation matrix as a pandas.DataFrame,
     - out_dir (Path, None): Output directory for saving the figure,
     - save_name (str, "corr.png"): The name of the output file,
     - check_allzeros (bool, True): Notify if all off-diagonal entries are all 0,
     - show (bool, False): Whether to display the figure.
     - lower_tri (bool, False): Return only the lower triangular matrix,
     - figsize (float 2-tuple, (25,8)): figure size in inches, (width, height).
     Note:
      If check_allzeros is True & the check returns True, there is no plotting.)
    """
    df_corr = df_corr.round(2)
    if check_allzeros:
        corr_array = df_corr.values
        # Create a mask for off-diagonal elements
        off_diag_mask = ~np.eye(corr_array.shape[0], dtype=bool)
        # Check if all off-diagonal elements are zero
        if np.all(corr_array[off_diag_mask] == 0):
            logging.warning("All off-diagonal correlation values are 0.00: not plotting.")
            return

    if df_corr.shape[0] > 14 and fig_size == HEATMAP_SIZE:
        logger.warning(("With a matrix size > 14 x 14, the fig_size argument" f" should be > {HEATMAP_SIZE}."))

    n_resample = 8
    top = mpl.colormaps["Reds_r"].resampled(n_resample)
    bottom = mpl.colormaps["Blues"].resampled(n_resample)
    newcolors = np.vstack((top(np.linspace(0, 1, n_resample)), bottom(np.linspace(0, 1, n_resample))))
    cmap = ListedColormap(newcolors, name="RB")
    norm = BoundaryNorm([-1.0, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1.0], cmap.N)

    if lower_tri:
        # mask to get lower triangular matrix w diagonal:
        msk = np.triu(np.ones_like(df_corr), k=1)
    else:
        msk = None

    fig = plt.figure(figsize=fig_size)

    fs = 12  # font size
    ax = sns.heatmap(
        df_corr,
        mask=msk,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        norm=norm,
        square=True,
        linecolor="white",
        linewidths=0.01,
        fmt=".2f",
        annot=True,
        annot_kws={"fontsize": 10},
    )
    ax.set(xlabel="", ylabel="")
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)  # rotation=90)
    plt.tight_layout()

    if save_name:
        if out_dir is not None:
            fig_fp = out_dir.joinpath(save_name)
        else:
            fig_fp = Path(save_name)

        plt.savefig(fig_fp, dpi=300, bbox_inches="tight")
        logger.info(f"Correlation heat map saved as {fig_fp}")

    if show:
        plt.show()
    else:
        plt.close()

    return
# <<< plotting functions - end ....................................


def cluster_corr_matrix(corr_df: pd.DataFrame, n_clusters: int = 5):
    """Return the clustered correlation matrix.
    Args:
      - corr_df (pd.DataFrame): correlation dataframe, i.e. df.corr();
      - n_clusters (int, 5): Number of candidate clusters, minimum 3;
    """
    # Convert correlation matrix to distance matrix
    dist_matrix = pdist(1 - np.abs(corr_df))

    # Perform hierarchical clustering
    linkage_matrix = linkage(dist_matrix, method="complete")  # "ward")

    if n_clusters < 3:
        n_clusters = 3

    clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    # Get the order of columns based on clustering
    ordered_cols = [corr_df.columns[i] for i in np.argsort(clusters)]

    # Return the reordered correlation matrix:
    return corr_df.loc[ordered_cols, ordered_cols]


def get_mcce_input_files(mcce_dir: str, ph_pt: str, eh_pt: str = "0") -> tuple:
    """Return the verified path to head3.lst and to the 'msout file'."""
    mcce_dir = Path(mcce_dir).resolve()
    h3_fp = mcce_dir.joinpath("head3.lst")
    if not h3_fp.exists():
        logger.error(f"FileNotFoundError: {str(h3_fp)} not found.")
        sys.exit(1)

    msout_fp = mcce_dir.joinpath("ms_out", f"pH{ph_pt}eH{eh_pt}ms.txt")
    if not msout_fp.exists():
        # try again with int:
        ph = int(ph_pt)
        eh = int(eh_pt)
        msout_fp = mcce_dir.joinpath("ms_out", f"pH{ph}eH{eh}ms.txt")
        if not msout_fp.exists():
            sys.exit(f"FileNotFoundError: {str(msout_fp)} not found")

    return h3_fp, msout_fp


def get_resid2iconf_dict(conformers: list) -> dict:
    """Return the mapping of conformer resid to its index."""
    return dict((c.resid, c.iconf) for c in conformers)


def check_res_list(correl_lst: list, res_lst: list = None, confs_lst: list = None) -> list:
    """Perform at most 2 checks on res_list depending on presence of the other
    list arguments:
     - Whether items in res_list are in other_list;
     - Whether items in res_list have a corresponding iconf.
    """
    if res_lst is None and confs_lst is None:
        logger.warning("Both 'res_lst' and 'confs_lst' arguments cannot be None.")
        return correl_lst

    if res_lst:
        new = []
        for res in correl_lst:
            if res[:3] in res_lst:
                new.append(res)
            else:
                logger.warning(f"Removing {res!r} from correl_lst: {res[:3]} not in residue_kinds.")
        correl_lst = new

    if not confs_lst:
        return correl_lst

    if confs_lst:
        correl2 = deepcopy(correl_lst)
        res2iconf = get_resid2iconf_dict(confs_lst)
        for cr in correl_lst:
            if res2iconf.get(cr) is None:
                logger.warning(f"Removing {cr!r} from correl_lst: not in conformers.")
                correl2.remove(cr)

        return correl2


def crg_msa_with_correlation(args1: dict, args2: dict):
    """Processing pipeline to obtain charge ms with correlation outputs.
    Args:
     - args1 (dict): general parameters.
     - args2 (dict): charge_histograms parameters
    """
    logger.info("Start msa")
    mcce_dir = Path(args1.get("mcce_dir", ".")).resolve()

    p, e = args1.get("msout_file", "pH7eH0ms.txt")[:-4].lower().split("eh")
    ph = p.removeprefix("ph")
    eh = e.removesuffix("ms")

    param_defaults = params_main(ph=ph)
    output_dir = mcce_dir.joinpath(args1.get("output_dir", param_defaults["output_dir"]))
    if not output_dir.exists():
        output_dir.mkdir()

    h3_fp, msout_fp = get_mcce_input_files(mcce_dir, ph, eh)

    logger.info(f"mcce_dir: {str(mcce_dir)}")
    logger.info(f"output_dir: {str(output_dir)}")
    logger.info(f"head3.lst: {str(h3_fp)}")
    logger.info(f"msout_file: {str(msout_fp)}")

    residue_kinds = args1.get("residue_kinds", IONIZABLES)
    choose_res = args1.get("correl_resids")

    conformers = read_conformers(h3_fp)
    logger.info(f"Number of conformers: {len(conformers):,}\n")

    if choose_res is None:
        logger.warning("No resids given for correlation.")
    else:
        # Check if ids kind in residue_kind and in conformers:
        correl_resids = check_res_list(choose_res, res_lst=residue_kinds, confs_lst=conformers)
        if not correl_resids:
            logger.warning("Empty 'correl_resids' post-validation for correlation.")
            choose_res = None
        elif len(correl_resids) < 2:
            logger.warning("Not enough 'correl_resids' left post-validation for correlation.")
            choose_res = None

    mc = MSout(msout_fp)
    logger.info(mc)

    # for all plots:
    show_fig = eval(args1.get("fig_show", "False"))

    # fixed res info:
    background_crg, all_fixed_res_crg_df, all_fixed_res_crg_dict = fixed_residues_info(mc.fixed_iconfs, conformers)
    logger.info(f"Background charge: {background_crg}")
    logger.info(f"Number of fixed residues: {len(all_fixed_res_crg_dict)}")

    # Save the free_residues in a pandas.DataFrame, it will be one of the inputs to the function msa.ConcaCrgMsPandas:
    free_residues_df = free_residues_to_df(mc.free_residues, conformers, colname="Residue")
    logger.info(f"Number of free residues: {free_residues_df.shape[0]:,}")

    # Get their net charges into a df for combining with fixed_res
    free_res_crg_df = free_res2sumcrg_df(mc.microstates.values(), mc.free_residues, conformers)

    # Combine free & fixed res with crg and save to csv:
    all_res_crg_df = combine_all_free_fixed_residues(free_res_crg_df, all_fixed_res_crg_df)
    all_res_crg_csv = args1.get("all_res_crg_csv", param_defaults["all_res_crg_csv"])
    all_res_crg_df.to_csv(output_dir.joinpath(all_res_crg_csv), index_label="Residue")

    # fixed res of interest info:
    background_crg, fixed_resoi_crg_df, fixed_resoi_crg_dict = fixed_residues_info(
        mc.fixed_iconfs, conformers, res_of_interest=residue_kinds
    )
    n_fixed_resoi = fixed_resoi_crg_df.shape[0]
    if n_fixed_resoi:
        logger.info(f"Fixed res in residues of interest: {n_fixed_resoi}")
        fixed_resoi_crg_df.to_csv(output_dir.joinpath(f"fixed_crg_resoi.csv"), index=False)
    else:
        fixed_resoi_crg_df = None
        logger.info("No fixed residues of interest.")

    # sorted ms list
    ms_orig_lst = [[ms.E, ms.count, ms.state] for ms in mc.sort_microstates()]

    # energies distribution plot:
    save_name = args1.get("energy_histogram.save_name", f"enthalpy_dist.png")
    fig_size = eval(args1.get("energy_histogram.fig_size", "(8,8)"))
    ms_energy_histogram(ms_orig_lst, output_dir, save_name, show=show_fig, fig_size=fig_size)

    id_vs_charge = iconf2crg(conformers)
    crg_orig_lst = ms2crgms(ms_orig_lst, id_vs_charge)

    # processing for histograms, inputs in args2 dict:
    for hist_d in list(args2.values()):
        plot_ok = True  # preset
        title = hist_d.get("title", "Charge Microstates Energy")
        bounds = hist_d.get("bounds")
        if bounds is None or "None" in bounds:
            ebounds = (None, None)
            df2csv_name = args1.get("main_csv", f"all_crg_count_res.csv")
            save_name = hist_d.get("save_name", f"crgms_logcount_v_E.png")
        elif "Emin" in bounds:
            offset = float(bounds[:-1].split("+")[1].strip())
            # use lowest E of the charge microstates:
            ebounds = (crg_orig_lst[0][0], crg_orig_lst[0][0] + offset)
            df2csv_name = f"low_crg_count_res.csv"
            save_name = hist_d.get("save_name", f"crgms_logcount_v_Emin.png")
        elif "Eaver" in bounds:
            # use average E of the conformer microstates:
            offset = float(bounds[:-1].split("+")[1].strip())
            ebounds = (mc.average_E - offset, mc.average_E + offset)
            df2csv_name = f"aver_crg_count_res.csv"
            save_name = hist_d.get("save_name", f"crgms_logcount_v_Eaver.png")
        elif "Emax" in bounds:
            # use max E of the conformer microstates:
            offset = float(bounds[1:].split("-")[1].split(",")[0].strip())
            ebounds = (mc.highest_E - offset, mc.highest_E)
            df2csv_name = f"high_crg_count_res.csv"
            save_name = hist_d.get("save_name", f"crgms_logcount_v_Emax.png")
        else:  # free bounds
            b, e = bounds[1:-1].strip().split(",")
            ebounds = (float(b.strip()), float(e.strip()))
            df2csv_name = f"range_crg_count_res.csv"
            save_name = hist_d.get("save_name", f"crgms_logcount_v_Erange.png")

        # compute:
        logger.info(f"Getting crgms data for energy bounds: {bounds}")
        crgms_info = find_uniq_crgms_count_order(crg_orig_lst, begin_energy=ebounds[0], end_energy=ebounds[1])
        plot_ok = len(crgms_info[0]) > 0
        crg_count_df = concat_crgms_dfs(
            crgms_info[0],
            crgms_info[1],
            crgms_info[2],
            free_residues_df,
            background_crg,
            res_of_interest=residue_kinds,
        )

        # save all crgms dfs:
        crg_count_df.to_csv(output_dir.joinpath(df2csv_name), header=True)

        if ebounds == (None, None):
            # preserve df for latter use:
            all_crg_count_res = crg_count_df.copy()

        # create figure
        if plot_ok:
            unique_crgms_histogram(
                crgms_info, background_crg, title, output_dir, save_name=save_name, show=show_fig
            )

    all_crg_df = add_fixed_res_crg(all_crg_count_res, fixed_resoi_crg_df)
    crg_count_csv = output_dir.joinpath(f"all_crg_count_resoi.csv")
    all_crg_df.to_csv(crg_count_csv, header=True)

    if choose_res is not None:
        logger.info("Computing correlation for chosen resids...")
        df_choose_res_data = choose_res_data(all_crg_df, correl_resids)
        df_choose_res_data["Occupancy"] = round(df_choose_res_data["Count"] / sum(df_choose_res_data["Count"]), 2)

        res_of_interest_data_csv = args1.get("res_of_interest_data_csv", f"crg_count_res_of_interest.csv")
        df_choose_res_data.to_csv(output_dir.joinpath(res_of_interest_data_csv), header=True)

        # Filter out fixed res from list:
        chosen_free = filter_fixed_from_choose_res(fixed_resoi_crg_df, choose_res)

        # Relabel residues with shorter names:
        df_chosen_res_renamed = rename_order_residues(df_choose_res_data, chosen_free)

        corr_method = args1.get("corr_method", "pearson")
        corr_cutoff = float(args1.get("corr_cutoff", 0))

        df_correlation = WeightedCorr(df=df_chosen_res_renamed, wcol="Count", cutoff=corr_cutoff)(
            method=corr_method
        )
        if df_correlation is not None:
            savename = args1.get("corr_heatmap.save_name", f"corr_heatmap_ph{ph}.png")
            figsize = eval(args1.get("corr_heatmap.fig_size", "(20, 8)"))
            clst = int(args1.get("n_clusters", "5"))
            clustered_corr = cluster_corr_matrix(df_correlation, n_clusters=clst)
            corr_heatmap(clustered_corr, out_dir=output_dir, save_name=savename, show=show_fig, fig_size=figsize)
    else:
        logger.info("No chosen resids: no correlation.")

    return


def crgmsa_parser() -> ArgumentParser:

    DESC = """cms_analysis_wc.py ::
Stand-alone charge microstate analysis with correlation.
Can be used without the mcce program & its codebase.
The only command line option is the parameters file.

CALL EXAMPLE:
  python cms_analysis_wc.py params.crgms

Note:
  If you add this line:
     list_head3_ionizables = true
  in your parameter file, the program will list the resids in head3.lst
  and exit; The list or a portion thereof can then be used as values to
  the 'correl_resids' identifier.
"""
    p = ArgumentParser(
        prog="cms analysis with correlation",
        description=DESC,
        formatter_class=RawDescriptionHelpFormatter,
    )
    p.add_argument("params_file", type=str, help="The input parameters file.")

    return p


def crgmsa_cli(argv=None):

    parser = crgmsa_parser()
    args = parser.parse_args(argv)

    params = Path(args.params_file)
    logger.info(f"params_file: {str(params)}")
    if not params.exists():
        sys.exit("Parameters file not found.")

    # load the parameters from the input file into 2 dicts:
    main_d, crg_histo_d = load_crgms_param(params)
    list_ionizables = main_d.get("list_head3_ionizables")
    if list_ionizables is not None:
        if list_ionizables.capitalize() == "True":
            logger.info("List of ionizable residues in head3.lst:")
            print(list_head3_ionizables(main_d.get("mcce_dir", ".")+"/head3.lst"))
            sys.exit()

    # run the processing pipeline:
    crg_msa_with_correlation(main_d, crg_histo_d)
    return


if __name__ == "__main__":
    crgmsa_cli(sys.argv[1:])
