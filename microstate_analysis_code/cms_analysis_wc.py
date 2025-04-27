#!/usr/bin/env python

"""
Module: cms_analysis_wc.py

  Stand-alone charge microstate analysis with correlation.
  Uses 'fast msout file loader' class MSout_np.

  Can be used without the mcce program & its codebase.
  Can be obtain from the codebase using wget
  TBD: give MCCE4 repo path once finalized.

COMMAND LINE CALL EXAMPLE. Parameter passing via a file:
  python cms_analysis_wc.py params.crgms
  OR
  ./cms_analysis_wc.py params.crgms

"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import defaultdict
from copy import deepcopy
from itertools import islice
import logging
from pathlib import Path
import re
import string
import sys
import time
from typing import Tuple, Union
import warnings

try:
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.pyplot as plt

    plt.ioff()
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    from scipy.stats import skewnorm, rankdata
except ImportError as e:
    print("[CRITICAL] Oops! Forgot to activate an appropriate environment?\n", e)
    sys.exit(1)

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
HEATMAP_SIZE = (20, 8)

# constants used by MSout_np:
MIN_OCC = 0.02  # occ threshold
N_top = 5  # top N default
# For translating HIS pseudo charges found in conf_info to tautomers:
HIS0_tautomers = {0: " NE2", 1: " ND1", 2: "+1"}


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


def reader_gen(fpath: Path):
    """
    Generator function yielding a file line.
    """
    with open(fpath) as fh:
        for line in fh:
            yield line


def show_elapsed_time(start_t: time, info: str = None):
    elapsed = time.time() - start_t
    if info is None:
        print(f"Elapsed time: {elapsed:,.2f} s ({elapsed/60:,.2f} min).")
    else:
        print(f"{info} - Elapsed time: {elapsed:,.2f} s ({elapsed/60:,.2f} min).")
    return

# FIX: REDO estimate
def topN_loadtime_estimate(n_freeres: int) -> str:
    """Returns the time estimate given the number of free residues
    for reading the mc lines to getting the topN ms in a formattted
    string showing seconds and minutes.
    """
    # fit of 5 runs:
    # -2 offset: improvements since fit
    return round(-14.9897855 -2 + 0.451883977*n_freeres + 6.25518650e-04*n_freeres**2)


class MSout_np:
    """Class to load 'msout file' to obtain MCCE microstates data in numpy.arrays.
    * Naming convention:
        - 'charge ms', 'crg ms' and 'cms' are shortcuts for protonation microstate.
        - 'ms' is a shortcut for conformational microstate.
        - 'msout file' refers to the .txt file in the ms_out subfolder of an mcce run
           and starts with 'pH<ph>'.

    Arguments:
        - head3_file, msout_file (str): Paths to head3.lst & msout files.
        - mc_load (str): Specifies what to load from the msout file:
           - 'all': crg and conf ms.
           - 'conf': conformation ms only (as in ms_analysis.MSout class);
           - 'crg': protonation ms only;
        - res_kinds (list): List of 3-letter residue/ligand names, e.g. ['GLU', 'HEM'];
          Defaults to IONIZABLES with mc_load='all' or 'crg'.

    Details:
        1. Reads head3.lst & the msout file header and creates a conformer data
           'lookup table' into np.array attribute 'conf_info'.
        2. Loads accepted states information from msout file MC lines into a list,
           which is recast to np.array at end of processing, yielding MSout_np.all_ms
           or MSout_np.all_cms.

    Example for obtaining topN data (crg ms and related conf ms):
        ```
        from msout_np import MSout_np

        h3_fp = "path/to/head3.lst"
        msout_fp = "path/to/ms_out/msout_file/filename"

        # using defaults: mc_load="all", res_kinds=IONIZABLES, loadtime_estimate=False
        mso = MSout_np(h3_fp, msout_fp)
        print(mso)

        # run get_uniq_ms() method, which will populate mso.uniq_cms (as per mc_load)
        mso.get_uniq_ms()
 
        # Get topN data using defaults: N = 5; min_occ = 0.0
        top5_cms, top5_ms = mso.get_topN_lists()
        ```
    """
    def __init__(self, head3_file: str, msout_file: str,
                 mc_load: str = "all",
                 res_kinds: list = None,
                 loadtime_estimate: bool = False,
                 ):

        # valid loading modes:
        loading_modes = ["conf", "crg", "all"]
        if mc_load.lower() not in loading_modes:
            print("Argument mc_load must be one of", loading_modes ,
                  "to load either conformer or charge microstates, or both.")
            return

        self.mc_load = mc_load.lower()
        self.msout_file = msout_file
        if res_kinds is None:
            if self.mc_load != "conf":
                self.res_kinds = IONIZABLES
        else:
            # TODO: test res_kinds with mc_load="conf"
            self.res_kinds = res_kinds

        # attributes populated by self.load_header:
        self.T: float = 298.15
        self.pH: float = 7.0
        self.Eh: float = 0.0
        self.fixed_iconfs: list = []
        self.free_residues: list = []  # needed in get_conf_info
        self.iconf2ires: dict = {}

        # attributes populated by self.get_conf_info:
        self.N_confs: int = None
        self.N_resid: int = None
        # conformer lookup table using head3 data:
        self.conf_info: np.ndarray = None
        self.conf_ids: np.ndarray = None
        # list of resids defining a cms:
        self.cms_resids: list = None
        # sum crg of fixed res:
        self.background_crg: int = None

        # attributes populated by the 'load' functions:
        self.N_space: int = None     # size of conformal state space
        self.N_mc_lines: int = None  # total number of mc lines (accepted states data)
        self.N_cms: int = None       # total number of crg (protonation) ms
        # np.arrays to receive the lists of conf/crg ms: 
        self.all_ms: np.ndarray = None
        self.all_cms: np.ndarray = None

        # attributes populated by the 'get_uniq_' functions:
        self.uniq_ms: np.ndarray = None
        self.uniq_cms: np.ndarray = None
        self.N_ms_uniq: int = None  # unique number of conf ms
        self.N_cms_uniq: int = None  # unique number of crg ms

        # load msout file header data:
        self.load_header()

        # create the head3 lookup array:
        self.conf_info, self.cms_resids, self.conf_ids = self.get_conf_info(head3_file)
        self.N_confs = len(self.conf_ids)
        self.N_resid = len(self.cms_resids)
        # fields :: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        # sumcrg for is_fixed:
        self.background_crg = self.conf_info[np.where(self.conf_info[:, -4]), -1].sum()

        # load accepted states:
        if self.mc_load == "crg":
            start_t = time.time()
            self.load_crg()
            show_elapsed_time(start_t, info="Loading msout for cms")
        elif self.mc_load == "conf":
            start_t = time.time()
            self.load_conf()
            show_elapsed_time(start_t, info="Loading msout for conf ms")
        elif self.mc_load == "all":
            if loadtime_estimate:
                yt = topN_loadtime_estimate(len(self.free_residues))
                print(f"\nESTIMATED TIME to topN: {yt:,.2f} s ({yt/60:,.2f} min).\n")
            start_t = time.time()
            self.load_all()
            show_elapsed_time(start_t, info="Loading msout for ms & cms")
        else:
            print("No processing function associated with:", self.mc_load)

        return

    def load_header(self):
        """Process an unadulterated 'msout file' header rows to populate
        these attributes: T, pH, Eh, fixed_iconfs, free_residues, iconf2ires.
        """
        with open(self.msout_file) as fh:
            head = list(islice(fh, 6))
        for i, line in enumerate(head, start=1):
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
                    print("This file %s is not a valid microstate file" % self.msout_file)
                    sys.exit(-1)
            if i == 4:
                _, iconfs = line.split(":")
                self.fixed_iconfs = [int(i) for i in iconfs.split()]
            if i == 6:  # free residues
                _, residues_str = line.split(":")
                residues = residues_str.split(";")
                for f in residues:
                    if f.strip():
                        self.free_residues.append([int(i) for i in f.split()])
                for idx, lst in enumerate(self.free_residues):
                    for iconf in lst:
                        self.iconf2ires[iconf] = idx
        return

    def get_conf_info(self, h3_fp: str) -> Tuple[np.ndarray, list, np.ndarray]:
        """Output these variables:
         - conf_info (np.ndarray): a lookup 'table' for iconfs, resids, and charges
           initially from head3.lst;
         - cms_resids (list): list of unique, free & ionizable resids in a MCCE simulation;
         - conf_ids (np.ndarray): array of iconfs, confids;

        Note:
        Final extended format of conf_info:
          [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg];
        Field 'is_fixed' is needed for proper accounting of net charge.
        """
        with open(h3_fp) as h3:
            lines = h3.readlines()[1:]

        conf_info = []
        conf_vec = []
        for line in lines:
            # ignored columns: FL, Occ & fields beyond crg
            iConf, confid, _, _, Crg, *_ = line.split()
            iconf = int(iConf) - 1  # as python index
            kind = confid[:3]
            resid = kind + confid[5:11]
            crg = int(float(Crg))
            if kind == "HIS":
                # get pseudo crg:
                # 0 :: HIS01->" NE2"; 1 :: HIS02->" ND1"; 2 :: HIS+1
                crg = int(confid[4]) - 1 if confid[3] == "0" else 2

            is_ioniz = int(resid[:3] in IONIZABLES)
            in_kinds = 1
            if self.res_kinds:
                in_kinds = int(kind in self.res_kinds)
            is_fixed = int(iconf in self.fixed_iconfs)
            # conf_info last 3 :: [..., is_free, resix, crg]
            conf_info.append([iconf, resid, in_kinds, is_ioniz, is_fixed, 0, 0, crg])
            conf_vec.append([iconf, confid])
        # temp list structure is now set & has h3 info; cast to np.ndarray:
        conf_info = np.array(conf_info, dtype=object)
        conf_ids = np.array(conf_vec, dtype=object)

        # update conf_info: use free iconfs from free_residues to
        # populate the 'is_free' field.
        free_iconfs = [ic for free in self.free_residues for ic in free]
        conf_info[free_iconfs, -3] = 1

        # get cms unique resids list via filtering conf_info for valid confs for
        # protonation state vec: ionizable & free & in user list if given.
        # conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        sum_conditions = conf_info[:, 3] + conf_info[:, -3]   # ionizable & free
        sum_tot = 2
        if self.res_kinds:
            # in_kinds & is_ioniz & is_free
            sum_conditions = conf_info[:, 2] + sum_conditions
            sum_tot = 3
        # Note: dict in use instead of a set (or np.unique) to preserve the order:
        d = defaultdict(int)
        for r in conf_info[np.where(sum_conditions == sum_tot)][:, 1]:
            d[r] += 1
        # uniq resids to list:
        cms_resids = list(d.keys())

        # create mapping from confs space to protonation resids space:
        # update conf_info resix field with the index from cms_resids list:
        # conf_info: [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]

        # Getting resix w/o checking again for is_free was not sufficient,
        # e.g. GLUA0007; iconfs 12, 13, 14 needed resix = -1, since not free:
        # [12 'GLUA0007_' 1 0 0 2 0]
        # [13 'GLUA0007_' 1 0 0 2 0]
        # [14 'GLUA0007_' 1 0 0 2 0]
        # [15 'GLUA0007_' 1 0 1 2 0]
        # [16 'GLUA0007_' 1 0 1 2 -1]
        for i, (_, resid, _, _, _, is_free, *_) in enumerate(conf_info):
            try:
                resix = cms_resids.index(resid)
                if not is_free:
                    resix = -1
            except ValueError:
                # put sentinel flag for unmatched res:
                resix = -1
            conf_info[i][-2] = resix

        print("\nHead3 lookup array 'conf_info'\n\tfields ::",
              "iconf:0, resid:1, in_kinds:2, is_ioniz:3,",
              "is_fixed:4, is_free:5, resix:6, crg:7\n")
        return conf_info, cms_resids, conf_ids

    def get_ter_dict(self) -> dict:
        """Return a dict for res with multiple entries, such as
        terminal residues.
        Sample output, dict: {'A0001': ['NTR', 'LYS'],
                              'A0129': ['LEU', 'CTR']}
        """
        ter_dict = defaultdict(list)
        for confid in self.conf_ids[:,1]:
            res = confid[:3]
            res_id = confid[5:].split("_")[0]
            # order needed, can't use set():
            if res not in ter_dict[res_id]:
                ter_dict[res_id].append(res)

        return dict((k, v) for k, v in ter_dict.items() if len(v) > 1)


    def load_conf(self):
        """Process the 'msout file' mc lines to populate a list of
        [state, state.E, count] items, where state is a list of conformal
        microstates.
        This list is then assignedd to MCout.all_ms as a numpy.array.
        """
        # print("Loading function: load_conf")
        found_mc = False
        newmc = False
        ms_vec = []  # list to hold conf ms info

        msout_data = reader_gen(self.msout_file)
        for lx, line in enumerate(msout_data):
            if lx < 9:
                continue
            line = line.strip()
            if not line or line[0] == "#":
                continue
            else:
                # find the next MC record
                if line.startswith("MC:"):
                    found_mc = True
                    newmc = True
                    continue

                if newmc:
                    # line with candidate state for MC sampling, e.g.:
                    # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free iconfs in state
                    state_e = 0.0
                    current_state = [int(i) for i in line.split(":")[1].split()]
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

                        ms_vec.append([list(current_state), state_e, count])
                    else:
                        continue

        self.all_ms = np.array(ms_vec, dtype=object)
        self.N_mc_lines = len(self.all_ms)
        print(f"Accepted states lines: {len(self.N_mc_lines):,}\n")
        self.N_space = self.all_ms[:, -1].sum()

        return

    def load_crg(self):
        """Process the accepted microstates lines to populate a list of
        [state, totE, averE, count] items, where state is a list of protonation
        microstates for the free & ionizable residues in the simulation.
        This list is then assignedd to MCout.all_cms as a numpy.array.
        """
        found_mc = False
        newmc = False
        ro = -1
        # list to hold crg ms info:
        cms_vec = []

        msout_data = reader_gen(self.msout_file)
        for lx, line in enumerate(msout_data):
            if lx < 7:
                continue
            line = line.strip()
            if not line or line[0] == "#":
                continue
            else:
                # find the next MC record, e.g. MC:4
                if line.startswith("MC:"):
                    found_mc = True
                    newmc = True
                    continue

                if newmc:
                    ro += 1  # will be 0 at "MC:0" + 1 line
                    # line with candidate state for MC sampling, e.g.:
                    # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free iconfs in state
                    state_e = 0.0
                    current_state = [int(i) for i in line.split(":")[1].split()]

                    # cms_vec :: [state, totE, averE, count]
                    cms_vec.append([[0] * len(self.cms_resids), 0, 0, 0])
                    # update cms_vec state:
                    curr_info = self.conf_info[current_state]

                    # acceptable conformer: ionizable, free, and in res_kinds if provided, meaning
                    # field 'resix' has a valid index (positive int) => resix != -1
                    # [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
                    upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                    for u in upd:
                        cms_vec[ro][0][u[0]] = u[1]
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

                        # flipped iconfs from non-ionizable or fixed res
                        # => same protonation state: increment totE & count;
                        # Note: -1 is a sentinel index for this situation.
                        update_cms = np.all(self.conf_info[flipped, -2] == -1)
                        if update_cms:
                            # cms_vec ::  [state, totE, averE, count]
                            cms_vec[ro][1] += state_e * count
                            cms_vec[ro][3] += count
                            cms_vec[ro][2] = cms_vec[ro][1] / cms_vec[ro][3]
                        else:
                            ro += 1
                            cms_vec.append([[0] * len(self.cms_resids), state_e * count, state_e, count])
                            curr_info = self.conf_info[current_state]
                            upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                            for u in upd:
                                cms_vec[ro][0][u[0]] = u[1]
                    else:
                        continue

        self.all_cms = np.array(cms_vec, dtype=object)
        self.N_space = self.all_cms[:, -1].sum()
        self.N_cms = len(self.all_cms)
        print(f"Protonation microstates: {len(self.N_cms):,}\n")

        return

    def load_all(self):
        """Process the 'msout file' mc lines to output both conformal
        and protonation microstates to numpy.arrays MSout_np.all_ms 
        and MSout_np.all_cms.
        """
        found_mc = False
        newmc = False
        ro = -1  # list item accessor
        # lists to hold conf and crg ms info; they can be related by their common index;
        cms_vec = []
        ms_vec = []

        msout_data = reader_gen(self.msout_file)
        for lx, line in enumerate(msout_data):
            if lx < 9:
                continue
            line = line.strip()
            if not line or line[0] == "#":
                continue
            else:
                # find the next MC record
                if line.startswith("MC:"):
                    found_mc = True
                    newmc = True
                    continue

                if newmc:
                    # line with candidate state for MC sampling, e.g.:
                    # 41:0 3 16 29 41 52 54 68 70 73 ... # ^N: number of free iconfs in state
                    state_e = 0.0
                    ro += 1  # will be 0 at "MC:0" + 1 line
                    current_state = [int(i) for i in line.split(":")[1].split()]

                    # cms_vec ::  [idx, state, totE, averE, count]
                    cms_vec.append([ro, [0] * len(self.cms_resids), 0, 0, 0])
                    # update cms_vec state:
                    curr_info = self.conf_info[current_state]
                    upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                    for u in upd:
                        cms_vec[ro][1][u[0]] = u[1]
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

                        ms_vec.append([ro, list(current_state), state_e, count])

                        # if the flipped iconfs are from non-ionizable or fixed res,
                        # the protonation state is the same: increment count & E;
                        # Note: -1 is a sentinel index for this situation.
                        update_cms = np.all(self.conf_info[flipped, -2] == -1)
                        if update_cms:
                            # cms_vec ::  [idx, state, totE, averE, count]
                            cms_vec[ro][2] += state_e * count
                            cms_vec[ro][4] += count
                            cms_vec[ro][3] = cms_vec[ro][2] / cms_vec[ro][4]
                        else:
                            ro += 1  # new cms
                            cms_vec.append([ro, [0] * len(self.cms_resids), state_e * count, state_e, count])

                            curr_info = self.conf_info[current_state]
                            upd = curr_info[np.where(curr_info[:, -2] != -1)][:, -2:]  # -> [resix, crg]
                            for u in upd:
                                cms_vec[ro][1][u[0]] = u[1]

        self.all_cms = np.array(cms_vec, dtype=object)
        self.N_cms = len(self.all_cms)
        self.all_ms = np.array(ms_vec, dtype=object)
        self.N_mc_lines = len(self.all_ms)
        print(f"Accepted states lines: {self.N_mc_lines:,}\n")
        print(f"Protonation microstates: {self.N_cms:,}\n")
        self.N_space = self.all_ms[:, -1].sum()

        return

    def get_uniq_ms(self):
        """Semaphore function to call the 'get unique' function corresponding
        to .mc_load loading mode.
        Note:
          If the loading mode is "all", it's assume that crg ms are the main
          focus thus, there is no need to get the unique conf ms as well.
          The equivalent function for conformers, `.get_uniq_all_ms()`
          is available if needed.
        """
        if self.mc_load == "conf":
            start_t = time.time()
            self.get_uniq_conf()
            show_elapsed_time(start_t, info="Populating .uniq_ms array")
            print(f"Unique conformer microstates: {self.N_cms_uniq:,}\n")

        elif self.mc_load == "crg":
            start_t = time.time()
            self.get_uniq_cms()
            show_elapsed_time(start_t, info="Populating .uniq_cms array")
            print(f"Unique protonation microstates: {self.N_cms_uniq:,}\n")

        elif self.mc_load == "all":
            start_t = time.time()
            self.get_uniq_all_cms()
            show_elapsed_time(start_t, info="Populating .uniq_cms array, 'all' mode")
            print(f"Unique protonation microstates: {self.N_cms_uniq:,}\n")

        else:
            print(f"WARNING: No processing function associated with: {self.mc_load}")

    def get_uniq_cms(self):
        """Assign unique crg ms info (state, totE, averE, count) to self.uniq_cms;
        Assign count of unique cms to self.N_cms_uniq.
        """
        subtot_d = {}
        # crg_e ::  [state, totE, averE, count]
        for ix, itm in enumerate(self.all_cms):
            key = tuple(itm[0])
            if key in subtot_d:
                subtot_d[key][1] += itm[1]
                subtot_d[key][3] += itm[3]
                subtot_d[key][2] = subtot_d[key][1] / subtot_d[key][3]
            else:
                subtot_d[key] = itm.copy()

        self.N_cms_uniq = len(subtot_d)
        # add occ, sort by count & assign to self.uniq_cms as np.array:
        # crg_e ::  [state, totE, averE, occ, count]
        self.uniq_cms = np.array(
            sorted(
                [
                    [list(k), subtot_d[k][1], subtot_d[k][2], subtot_d[k][3] / self.N_space, subtot_d[k][3]]
                    for k in subtot_d
                ],
                key=lambda x: x[-1],
                reverse=True,
            ),
            dtype=object,
        )

        return

    def get_uniq_conf(self):
        """Assign unique conf ms info (state, stateE, occ, count) to self.uniq_ms;
        Assign count of unique ms to self.N_ms_uniq.
        """
        if self.mc_load != "conf":
            print("WARNING: Redirecting to 'get_uniq_all_ms' function as per 'mc_load'.")
            self.get_uniq_all_ms()
            return
        # ms in ::  [state, state.e, count]
        subtot_d = {}
        for _, itm in enumerate(self.all_ms):
            key = tuple(itm[0])
            if key in subtot_d:
                subtot_d[key][2] += itm[2]
            else:
                subtot_d[key] = itm.copy()

        self.N_ms_uniq = len(subtot_d)
        # add occ, sort by count & assign to self.uniq_ms as np.array:
        # ms out ::  [state, state.e, occ, count]
        mslist = [
            [list(k), subtot_d[k][1], subtot_d[k][-1] / self.N_space, subtot_d[k][-1]] for k in subtot_d
        ]
        self.uniq_ms = np.array(sorted(mslist, key=lambda x: x[-1], reverse=True), dtype=object)
        return

    def get_uniq_all_cms(self):
        """Get the unique charge ms array when the `all_cms` array
        was produced together with the `all_ms` array, i.e. mc_load='all'.
        In this case, each of their items starts with an index,
        which can be used to match conf ms to each unique cms.
        """
        subtot_d = {}
        # vec :: [idx, state, totE, averE, count]
        for ix, itm in enumerate(self.all_cms):
            key = tuple(itm[1])
            if key in subtot_d:
                subtot_d[key][2] += itm[2]
                subtot_d[key][4] += itm[4]
                subtot_d[key][3] = subtot_d[key][2] / subtot_d[key][4]
            else:
                subtot_d[key] = itm.copy()

        self.N_cms_uniq = len(subtot_d)
        print(f"Unique protonation microstates: {self.N_cms_uniq:,}")

        # add occ, sort by count & assign to self.uniq_cms as np.array:
        # crg ms ::  [idx, state, totE, averE, occ, count]
        self.uniq_cms = np.array(
            sorted(
                [
                    [
                        subtot_d[k][0],
                        list(k),
                        subtot_d[k][2],
                        subtot_d[k][3],
                        subtot_d[k][4] / self.N_space,
                        subtot_d[k][4],
                    ]
                    for k in subtot_d
                ],
                key=lambda x: x[-1],
                reverse=True,
            ),
            dtype=object,
        )

        return

    def get_uniq_all_ms(self):
        # ms in ::  [idx, state, state.e, count]
        subtot_d = {}
        for _, itm in enumerate(self.all_ms):
            key = tuple(itm[1])
            if key in subtot_d:
                subtot_d[key][3] += itm[3]
            else:
                subtot_d[key] = itm.copy()

        self.N_ms_uniq = len(subtot_d)
        # add occ, sort by count & assign to self.uniq_ms as np.array:
        # ms out ::  [idx, state, state.e, occ, count]
        mslist = [
            [subtot_d[k][0], list(k), subtot_d[k][2], subtot_d[k][-1] / self.N_space, subtot_d[k][-1]]
            for k in subtot_d
        ]
        self.uniq_ms = np.array(sorted(mslist, key=lambda x: x[-1], reverse=True), dtype=object)
        return
    
    def get_free_residues_df(self) -> pd.DataFrame:
        """Extract resid for is_free from lookup array into a pandas.DataFrame."""
        free_residues_df = pd.DataFrame(self.conf_info[np.where(self.conf_info[:, -3]), 1][0],
                                        columns=["Residue"])
        free_residues_df.drop_duplicates(inplace=True)
        return free_residues_df

    def get_fixed_residues_arr(self) -> np.ndarray:
        """Extract resid, crg for is_ioniz & is_fixed from lookup array."""
        # [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        is_ioniz_fixed = self.conf_info[np.where(np.logical_and(self.conf_info[:, 3],
                                                                self.conf_info[:, 4]))]
        return is_ioniz_fixed[:, [1, -1]]

    def get_fixed_residues_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.get_fixed_residues_arr(), columns=["Residue", "crg"])

    def get_fixed_res_of_interest_arr(self) -> np.ndarray:
        """Extract resid, crg for in_kinds & is_fixed from lookup array."""
        # [iconf, resid, in_kinds, is_ioniz, is_fixed, is_free, resix, crg]
        return self.conf_info[np.where(np.logical_and(self.conf_info[:, 2],
                                                      self.conf_info[:, 4]))][:, [1, -1]]

    def get_fixed_res_of_interest_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.get_fixed_res_of_interest_arr(),
                            columns=["Residue", "crg"])

    def get_cms_energy_stats(self) -> Tuple[float, float, float]:
        """Return the minimum, average, and maximum energies of the cms in .all_cms."""
        cms_e = self.all_cms[:, -2]
        return round(np.min(cms_e), 2), round(np.mean(cms_e), 2), round(np.max(cms_e), 2)

    def get_ms_energy_stats(self) -> Tuple[float, float, float]:
        """Return the minimum, average, and maximum energies of the conf ms in .all_ms."""
        ms_e = self.all_ms[:, -2]
        return round(np.min(ms_e), 2), round(np.mean(ms_e), 2), round(np.max(ms_e), 2)

    @staticmethod
    def filter_cms_E_within_bounds(top_cms: Union[list, np.ndarray],
                                   E_bounds: Tuple[float, float]) -> list:
        """
        Filter top_cms for cms with energies within E_bounds.
        """
        if E_bounds == (None, None):
            return top_cms

        # index of energy data:
        E = 3 if len(top_cms[0]) == 6 else 2   # 2 :: array from mc_load=="crg"
        filtered = []
        for i, ro in enumerate(top_cms[:, 0]):
            # ignore top cms out of bounds:
            if top_cms[i, E] < E_bounds[0] or top_cms[i, E] > E_bounds[1]:
                continue
            filtered.append(top_cms[i])

        return filtered

    def get_topN_data(self, N: int = 5, min_occ: float = MIN_OCC, all_ms_out: bool = False) -> Tuple[list, dict]:
        """Return a 2-tuple:
         - top_cms (list): Containing the first N most numerous cms array with occ >= min_occ;
           NOTE: HIS charges are pseudo charges, see .get_conf_info for details.
         - top_ms (dict): Dict with key as the index shared by the .all_cms and .all_ms arrays
           (integer at position 0);
           The dict values depend on 'all_ms_out': if False (default), they are the most numerous
           conformer ms for the given index, else they are all the associated ms for that index.

        NOTE: Both outputs will be empty if all uniq_cms's occupancies are below the threshold.

        Call examples:
          1. Using defaults: dict top_ms values are a single related conformer ms:
            > top_cms, top_ms_dict = msout_np.get_topN_data()
          2. With 'all_ms_out' set to True: dict values are all related conf ms:
            > top_cms, top_ms_dict = msout_np.get_topN_data(all_ms_out=True)
        """
        if self.mc_load != "all":
            print("CRITICAL: Function '.get_topN_data' returns the topN cms along with the",
                  "associated conformer ms,\nso argument 'mc_load' must be 'all', not", self.mc_load)
            sys.exit(1)

        # recast input as they can be strings via the cli:
        N = int(N)
        min_occ = float(min_occ)
        print(f"Processing all_cms & all_ms for requested top {N} at {min_occ = :.1%}")

        if N > self.N_cms_uniq:
            N = self.N_cms_uniq
            print("Requested topN greater than available: processing all.")
        top_cms = []
        top_ms = {}
        topN_cms = self.uniq_cms[:N]
        for i, ro in enumerate(topN_cms[:, 0]):
            # ignore top cms with occ below threshold:
            if topN_cms[i, -2] < min_occ:
                continue
            top_cms.append(topN_cms[i])
            # get the associated conf ms:
            matched_ms = self.all_ms[np.where(self.all_ms[:, 0] == ro)]
            if all_ms_out:
                top_ms[ro] = matched_ms
            else:
                if len(matched_ms) > 1:
                    # get the most numerous:
                    top_ms[ro] = sorted(matched_ms, key=lambda x: x[-1], reverse=True)[0]
                else:
                    top_ms[ro] = matched_ms[0]
        if top_cms:
            print(f"Number of top cms returned: {len(top_cms):,}")
        return top_cms, top_ms

    def top_cms_df(self, top_cms: list,
                   output_tauto: bool = True,
                   cms_wc_format: bool = False) -> pd.DataFrame:
        """
        Arguments:
          - output_tauto: Set to False to keep the charge instead of the string tautomer.
          - cms_wc_format: Set to True to get df formatted for crg ms analysis with
                           weighted correlation.
        """
        fixed_free_res = None
        n_ffres = 0
        if not cms_wc_format:
            fixed_free_res = self.get_fixed_res_of_interest_arr()
            n_ffres = len(fixed_free_res) 
            if not n_ffres:
                fixed_free_res = None

        data = []
        for itm in top_cms:
            # [ idx, list(state), totE, averE, occ, count ]
            #     0        1       2      3     4     5
            fields = [itm[0]]   # the shared index
            state = itm[1].copy()
            for i, s in enumerate(state):
                if self.cms_resids[i][:3] == "HIS":
                    if output_tauto:
                        s = HIS0_tautomers[s]
                    else:
                        if s > 0:
                            state[i] -= 1
                            s -= 1
                fields.extend([s])
            if fixed_free_res is not None:
                fields.extend(fixed_free_res[:,1])
            fields.extend([round(itm[3], 2), sum(state) + self.background_crg, itm[5], round(itm[4], 4)])
            data.append(fields)

        if not cms_wc_format:
            res_cols = self.cms_resids

            if fixed_free_res is not None:
                # add fixed ionizable res
                res_cols = res_cols + fixed_free_res[:,0].tolist() 
                info_dat = ["tmp"] + ["free"] * len(self.cms_resids) + ["fixed"] * n_ffres + ["totals"] * 4
            else:
                # order as in data
                info_dat = ["tmp"] + ["free"] * len(self.cms_resids) + ["totals"] * 4

            # always remove trailing underscore
            res_cols = [c.rstrip("_") for c in res_cols]
            cols = ["idx"] + res_cols +  ["E", "sum_crg", "size", "occ"]
       
            df = pd.DataFrame(data, columns=cols)
            df["res"] = df.index + 1
            df.set_index("res", inplace=True)
            df = df.T
            df.columns.name = ""
            df.reset_index(inplace=True)
            df.rename(columns={"index":"residues"}, inplace=True)
            df["info"] = info_dat
            
            return df
        
        # Format as in microstate_analysis_code/cms_analysis_wc.py of https://github.com/Raihancuny/python/
        cols = ["Order"] + self.cms_resids + ["E", "SumCharge", "Count", "Occupancy"]
        df = pd.DataFrame(data, columns=cols)
        df.columns.name = "Residue"
        df.drop("E", axis=1, inplace=True)
        df["Order"] = df.index + 1

        # move SumCharge to end:
        new_cols = df.columns[:-3].tolist() + ["Count", "Occupancy", "SumCharge"]

        return df[new_cols].set_index("Order")

    def get_sampled_cms(self, size: int, seed: int = None) -> Union[np.ndarray, None]:
        """
        Return a random sample of crg microstates in a numpy.array, or None if MSout_np was
        intanciated with 'mc_load', the class 'loading mode', set to 'conf' as no crg ms are
        loaded in that case.
        Args:
            size (int): Sample size
            seed (int, None): seed for random number generator; pass an integer for reproducibility
        """
        if self.mc_load == "conf":
            print("Not applicable: 'get_sampled_cms' returns sampled crg microstates, but", 
                  "MSout_np was instantiated with mc_load='conf'; 'mc_load' must be 'crg' or 'all'.")
            return None
        rng = np.random.default_rng(seed=seed)
        indices = rng.integers(low=0, high=self.N_mc_lines, size=size, endpoint=True)
        return self.all_cms[indices].copy()

    def get_sampled_ms(self, size: int, seed: int = None) -> Union[np.ndarray, None]:
        """
        Return a random sample of conformer microstates in a numpy.array, or None if MSout_np was
        intanciated with 'mc_load', the class 'loading mode', set to 'crg' as no conf ms are
        loaded in that case.
        Args:
            size (int): Sample size
            seed (int, None): seed for random number generator; pass an integer for reproducibility
        """
        if self.mc_load == "crg":
            print("Not applicable: 'get_sampled_ms' returns sampled conformer microstates, but", 
                  "MSout_np was instantiated with mc_load='crg'; 'mc_load' must be 'conf' or 'all'.")
            return None
        rng = np.random.default_rng(seed=seed)
        indices = rng.integers(low=0, high=self.N_mc_lines, size=size, endpoint=True)
        return self.all_ms[indices].copy()

    def get_free_res_aver_crg_df(self) -> pd.DataFrame:
        """Convert the conformer ms state in MSout_np.all_ms collection to crg
        and return the average charge of all free residue into a pandas.DataFrame.
        """
        charges_total = defaultdict(float)
        ix_state = 1 if self.mc_load == "all" else 0

        for _, ms in enumerate(self.all_ms):
            for _, (resid, crg) in enumerate(self.conf_info[ms[ix_state]][:, [1, -1]]):
                charges_total[resid] += crg * ms[-1]

        return pd.DataFrame(
            [(k, round(charges_total[k] / self.N_space)) for k in charges_total], columns=["Residue", "crg"]
        )

    def __str__(self):
        return (f"\nConformers: {self.N_confs:,}\n"
                f"Conformational state space: {self.N_space:,}\n"
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

            if not (
                (isinstance(xyw, pd.DataFrame)) != (np.all([isinstance(i, pd.Series) for i in [x, y, w]]))
            ):
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
        cor = self._pearson
        if method == "spearman":
            cor = self._spearman

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
        "msout_file": f"pH{ph}eH0ms.txt",
        # Do not output file 'all_crg_count_res_ph7.csv'
        # "main_csv": "all_crg_count_res_ph7.csv",
        "all_res_crg_csv": "all_res_crg_status.csv",
        "res_of_interest_data_csv": "crg_count_res_of_interest.csv",
        "n_top": str(N_top),
        "min_occ": str(MIN_OCC),
        "residue_kinds": IONIZABLES,
        "correl_resids": None,
        "corr_method": "pearson",
        "corr_cutoff": "0.02",
        "n_clusters": "5",
        "fig_show": "False",
        "energy_histogram.save_name": "enthalpy_dist.png",
        "energy_histogram.fig_size": "(8,8)",
        "corr_heatmap.save_name": "corr.png",
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
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith("#")]

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
                        val = [
                            v for v in split_spunct(rawval[1:-1].strip(), pattern=correl_split_pattern) if v
                        ]
                else:
                    if key == "residue_kinds":
                        sys.exit(
                            (
                                "Malformed residue_kinds entry, ']' not found: "
                                "list within square brackets must be on the same line."
                            )
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


def choose_res_data(top_df: pd.DataFrame, correl_resids: list) -> pd.DataFrame:
    df = top_df.copy()
    out_cols = correl_resids + df.columns[-3:-1].tolist()
    df = df[out_cols]
    return df.reset_index(drop=True)


def add_fixed_resoi_crg_to_topdf(
    top_df: pd.DataFrame, fixed_resoi_crg_df: pd.DataFrame, cms_wc_format: bool = False
) -> pd.DataFrame:
    n_ro = top_df.shape[0]
    all_res_crg_df = top_df.copy()
    resois = fixed_resoi_crg_df.Residue.tolist()
    if cms_wc_format:
        totals_col1_idx = 3
    else:
        totals_col1_idx = 4
    out_cols = (
        all_res_crg_df.columns[:-totals_col1_idx].tolist() + resois + all_res_crg_df.columns[-3:].tolist()
    )

    for i, res in enumerate(resois):
        all_res_crg_df[res] = fixed_resoi_crg_df.iloc[i, 1] * n_ro

    return all_res_crg_df[out_cols]


def rename_reorder_df_cols(choose_res_data_df: pd.DataFrame) -> pd.DataFrame:
    """Output a new df with resids in res_cols with this format:
    chain + 1-letter res code + seq num,  and with this group order:
    acid, polar, base, ub_q, non_res kinds.
    """
    res_cols = choose_res_data_df.columns[0:-2].tolist()
    # termini
    ter_cols = []
    for i, col in enumerate(choose_res_data_df.columns[0:-2].tolist()):
        if col[:3].startswith(("NTR", "CTR")):
            ter_cols.append(res_cols.pop(i))

    res_cols = res_cols + ter_cols
    choose_res_data_df = choose_res_data_df[res_cols + ["Count"]]
    res_cols = choose_res_data_df.columns[0:-1].tolist()

    col_order = defaultdict(list)
    mapping = {}
    for res in res_cols:
        r3 = res[:3]
        rout = f"{res[3]}{res3_to_res1.get(r3, r3)}{int(res[4:-1])}"
        mapping[res] = rout
        if r3 in ACIDIC_RES:
            col_order[1].append((rout, ACIDIC_RES.index(r3)))
        elif r3 in POLAR_RES:
            col_order[2].append((rout, POLAR_RES.index(r3)))
        elif r3 in BASIC_RES:
            col_order[3].append((rout, BASIC_RES.index(r3)))
        elif r3 == "PL9":
            col_order[4].append((rout, 8888))
        else:
            col_order[5].append((rout, 9999))
    new_order = [v[0] for k in col_order for v in sorted(col_order[k], key=lambda x: x[1])]

    return choose_res_data_df.rename(columns=mapping)[new_order + ["Count"]]


def combine_all_free_fixed_residues(
    free_res_crg_df: pd.DataFrame, fixed_res_crg_df: pd.DataFrame
) -> pd.DataFrame:
    free_res_crg_df["status"] = "free"
    fixed_res_crg_df["status"] = "fixed"
    df = pd.concat([free_res_crg_df, fixed_res_crg_df])
    df.set_index("Residue", inplace=True)
    return df.T


# >>> plotting functions ....................................
def ms_energy_distribution(
    all_ms: np.ndarray,
    out_dir: Path,
    save_name: str = "enthalpy_dist.png",
    show: bool = False,
    fig_size=(8, 8),
):
    """
    Plot the histogram and distribution fit of the conformer microstates energies.
    Arguments:
      - all_ms (np.ndarray): Array of conformer microstates data returned by MSout_np.
    """
    energies = np.array(sorted(all_ms[:, -2]), dtype=float)
    skewness, mean, std = skewnorm.fit(energies)

    fig = plt.figure(figsize=fig_size)
    fs = 12  # fontsize

    graph_hist = plt.hist(energies, bins=100, alpha=0.6)
    Y = graph_hist[0]
    y = skewnorm.pdf(energies, skewness, mean, std)

    pdf_data = Y.max() / max(y) * y
    plt.plot(energies, pdf_data, label="approximated skewnorm", color="k")
    plt.title(f"{skewness = :.2f} {mean = :.2f} {std = :.2f}", fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.xlabel("Microstate Energy (Kcal/mol)", fontsize=fs)
    plt.ylabel("Count", fontsize=fs)
    plt.tick_params(axis="x", direction="out", length=8, width=2)
    plt.tick_params(axis="y", direction="out", length=8, width=2)

    fig_fp = out_dir.joinpath(save_name)
    fig.savefig(fig_fp, dpi=300, bbox_inches="tight")
    print(f"Microstate energy distribution figure saved: {fig_fp!s}")

    if show:
        plt.show()
    else:
        plt.close()

    return


def crgms_energy_histogram(
    top_cms: Union[list, np.ndarray],
    background_crg: int,
    fig_title: str,
    out_dir: Path,
    save_name: str,
    show: bool = False,
):
    """Plot charge microstates average energies vs the state protein charge,
    along with a marginal histogram.
    """
    data = np.array([[sum(arr[1]) + background_crg, arr[-1]] for arr in top_cms])
    net_crg = data[:, 0]

    fs = 12  # font size for axes labels and title
    
    g1 = sns.JointGrid(marginal_ticks=True, height=6)
    ax = sns.scatterplot(
        x=net_crg,
        y=data[:, 1],
        size=data[:, 1],
        legend="brief",
        ax=g1.ax_joint,
    )
    plt.yscale("log")

    ax.set_xticks(range(int(min(net_crg)), int(max(net_crg)) + 1))
    ax.set_xlabel("Charge", fontsize=fs)
    ax.set_ylabel("log$_{10}$(Count)", fontsize=fs)
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc=2, borderaxespad=0.0)

    ax2 = sns.histplot(x=net_crg, linewidth=2, discrete=True, ax=g1.ax_marg_x)
    ax2.set_ylabel(None, fontsize=fs)
    g1.ax_marg_y.set_axis_off()
    g1.fig.subplots_adjust(top=0.9)
    if fig_title:
        g1.fig.suptitle(fig_title, fontsize=fs)
    fig_fp = out_dir.joinpath(save_name)
    g1.savefig(fig_fp, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {fig_fp}")

    if show:
        plt.show()
    else:
        plt.close()

    return


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
     - check_allzeros (bool, True): Notify if all off-diagonal entries are 0,
     - show (bool, False): Whether to display the figure.
     - lower_tri (bool, False): Return only the lower triangular matrix,
     - figsize (float 2-tuple, (25,8)): figure size in inches, (width, height).
     Note:
      If check_allzeros is True & the check returns True, there is no plotting.
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
        logger.warning(
            ("With a matrix size > 14 x 14, the fig_size argument" f" should be > {HEATMAP_SIZE}.")
        )

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

        fig.savefig(fig_fp, dpi=300, bbox_inches="tight")
        logger.info(f"Correlation heat map saved as {fig_fp}")

    if show:
        plt.show()
    else:
        plt.close()

    return


# <<< plotting functions - end ....................................


def cluster_corr_matrix(corr_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Return the clustered correlation matrix.
    Args:
      - corr_df (pd.DataFrame): correlation dataframe, i.e. df.corr();
      - n_clusters (int, 5): Number of candidate clusters, minimum 3;
    """
    # Convert correlation matrix to distance matrix
    dist_matrix = pdist(1 - np.abs(corr_df))

    # Perform hierarchical clustering
    linkage_matrix = linkage(dist_matrix, method="complete")

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


def get_resid2iconf_dict(conf_info: np.ndarray) -> dict:
    """Return the mapping of conformer resid to its index.
    Note: Not to be used except for checking the existence of a resid
    in the conformer space.
    """
    free_ci = conf_info[np.where(conf_info[:, -3])]
    return dict((ci[0], ci[1]) for ci in free_ci[:, [1, 0]])


def check_res_list(correl_lst: list, res_lst: list = None, conf_info: np.ndarray = None) -> list:
    """Perform at most 2 checks on res_list depending on presence of the other arguments:
    - Whether items in res_list are in other_list;
    - Whether items in res_list are in the conformer space.
    """
    if not res_lst and not conf_info:
        logger.warning("Arguments 'conf_info' and 'res_lst' cannot both be None.")
        return correl_lst

    new = []
    for res in correl_lst:
        if res[:3] in res_lst:
            new.append(res)
        else:
            logger.warning(f"Ignoring {res!r} from correl_lst: {res[:3]} not in residue_kinds.")
    correl_lst = new

    if conf_info is not None:
        correl2 = deepcopy(correl_lst)
        res2iconf = get_resid2iconf_dict(conf_info)
        for cr in correl_lst:
            if res2iconf.get(cr) is None:
                logger.warning(f"Removing {cr!r} from correl_lst: not in conformer space.")
                correl2.remove(cr)

        return correl2

    return correl_lst


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
    if set(residue_kinds).difference(IONIZABLES):
        residue_kinds = sort_resoi_list(residue_kinds)
    choose_res = args1.get("correl_resids")

    # instantiate the 'fast loader' class:
    mc = MSout_np(h3_fp, msout_fp, res_kinds=residue_kinds)
    logger.info(mc)

    if choose_res is None:
        logger.warning("No resids given for correlation.")
    else:
        # Check if ids kind in residue_kind and in conformers:
        correl_resids = check_res_list(choose_res, res_lst=residue_kinds, conf_info=mc.conf_info)
        if not correl_resids:
            logger.warning("Empty 'correl_resids' post-validation for correlation.")
            choose_res = None
        elif len(correl_resids) < 2:
            logger.warning("Not enough 'correl_resids' left post-validation for correlation.")
            choose_res = None

    # for all plots:
    show_fig = eval(args1.get("fig_show", "False"))

    # fixed res info:
    all_fixed_res_crg_df = mc.get_fixed_residues_df()
    # free res average crg
    free_res_aver_crg_df = mc.get_free_res_aver_crg_df()

    # Combine free aver crg & fixed res with crg and save to csv:
    all_res_crg_df = combine_all_free_fixed_residues(free_res_aver_crg_df, all_fixed_res_crg_df)

    csv_fp = output_dir.joinpath(args1.get("all_res_crg_csv", param_defaults["all_res_crg_csv"]))
    msg = (
        f"Saving all_res_crg_df to {csv_fp!s}.\n",
        "Note: For residues with 'free' status, the charge is the average charge.",
    )
    logger.info(msg)
    all_res_crg_df.to_csv(csv_fp)

    # fixed res of interest info:
    fixed_resoi_crg_df = mc.get_fixed_res_of_interest_df()
    n_fixed_resoi = fixed_resoi_crg_df.shape[0]
    if n_fixed_resoi:
        csv_fp = output_dir.joinpath(
            args1.get("res_of_interest_data_csv", param_defaults["res_of_interest_data_csv"])
        )
        msg = (
            f"Fixed res in residues of interest: {n_fixed_resoi}\n"
            f"Saving fixed_resoi_crg_df to {csv_fp!s}.\n"
        )
        logger.info(msg)
        fixed_resoi_crg_df.to_csv(csv_fp, index=False)
    else:
        fixed_resoi_crg_df = None
        logger.info("No fixed residues of interest.")

    # energies distribution plot:
    save_name = output_dir.joinpath(
        args1.get("energy_histogram.save_name", param_defaults["energy_histogram.save_name"])
    )
    fig_size = eval(args1.get("energy_histogram.fig_size", "(8,8)"))
    ms_energy_distribution(mc.all_ms, output_dir, save_name, show=show_fig, fig_size=fig_size)

    # get unique crg ms:
    n_top = int(args1.get("n_top", param_defaults["n_top"]))
    min_occ = float(args1.get("min_occ", param_defaults["min_occ"]))

    top_cms, top_ms = mc.get_topN_lists(N=n_top, min_occ=min_occ)
    top_df = mc.top_cms_df(top_cms, output_tauto=False, cms_wc_format=True)
    top_df.to_csv(output_dir.joinpath("all_crg_count_res.csv"), header=True)

    all_res_crg_df = add_fixed_resoi_crg_to_topdf(top_df, fixed_resoi_crg_df, cms_wc_format=True)
    all_res_crg_df.to_csv(output_dir.joinpath("all_crg_count_resoi.csv"), header=True)

    # processing for histograms, inputs in args2 dict:
    # get min, aver, max E of crg ms
    cms_E_stats = mc.get_cms_energy_stats()

    for hist_d in list(args2.values()):
        title = hist_d.get("title", "Protonation Microstates Energy")
        bounds = hist_d.get("bounds")
        if bounds is None or "None" in bounds:
            ebounds = (None, None)
            # df2csv_name = args1.get("main_csv", "all_crg_count_res.csv")
            save_name = hist_d.get("save_name", "crgms_logcount_v_E.png")
        elif "Emin" in bounds:
            offset = float(bounds[:-1].split("+")[1].strip())
            # use lowest E of the charge microstates:
            ebounds = (cms_E_stats[0], cms_E_stats[0] + offset)
            save_name = hist_d.get("save_name", "crgms_logcount_v_Emin.png")
        elif "Eaver" in bounds:
            # use average E of the conformer microstates:
            offset = float(bounds[:-1].split("+")[1].strip())
            ebounds = (cms_E_stats[1] - offset, cms_E_stats[1] + offset)
            save_name = hist_d.get("save_name", "crgms_logcount_v_Eaver.png")
        elif "Emax" in bounds:
            # use max E of the conformer microstates:
            offset = float(bounds[1:].split("-")[1].split(",")[0].strip())
            ebounds = (cms_E_stats[2] - offset, cms_E_stats[2])
            save_name = hist_d.get("save_name", "crgms_logcount_v_Emax.png")
        else:  # free bounds
            b, e = bounds[1:-1].strip().split(",")
            ebounds = (float(b.strip()), float(e.strip()))
            save_name = hist_d.get("save_name", "crgms_logcount_v_Erange.png")

        # compute:
        logger.info(f"Filtering crgms data for energy bounds: {bounds}")
        filtered_cms = mc.filter_cms_E_within_bounds(mc.all_cms, ebounds)
        if len(filtered_cms):
            crgms_energy_histogram(
                filtered_cms, mc.background_crg, title, output_dir, save_name=save_name, show=False
            )

    if choose_res is not None:
        logger.info("Computing correlation for chosen resids...")
        choose_res_data_df = choose_res_data(top_df, correl_resids)
        res_of_interest_data_csv = args1.get("res_of_interest_data_csv", "crg_count_res_of_interest.csv")
        choose_res_data_df.to_csv(output_dir.joinpath(res_of_interest_data_csv), header=True)

        # Relabel residues with shorter names:
        df_chosen_res_renamed = rename_reorder_df_cols(choose_res_data_df)
        if df_chosen_res_renamed.shape[0] < 2:
            logger.info("Too few rows for correlation.")
            # done
            return

        corr_method = args1.get("corr_method", "pearson")
        corr_cutoff = float(args1.get("corr_cutoff", 0))

        df_correlation = WeightedCorr(df=df_chosen_res_renamed, wcol="Count", cutoff=corr_cutoff)(
            method=corr_method
        )
        if df_correlation is not None:
            savename = args1.get("corr_heatmap.save_name", "corr_heatmap.png")
            figsize = eval(args1.get("corr_heatmap.fig_size", "(20, 8)"))
            if df_correlation.shape[0] > 6:
                clst = int(args1.get("n_clusters", "5"))
                msg = f"Clustering the correlation matrix with at most {clst} clusters."
                logger.info("Clustering the correlation matrix")
                clustered_corr = cluster_corr_matrix(df_correlation, n_clusters=clst)
                corr_heatmap(
                    clustered_corr, out_dir=output_dir, save_name=savename, show=show_fig, fig_size=figsize
                )
            else:
                corr_heatmap(
                    df_correlation, out_dir=output_dir, save_name=savename, show=show_fig, fig_size=figsize
                )
    else:
        logger.info("No chosen resids: no correlation.")

    return


def list_head3_ionizables(h3_fp: Path, as_string: bool = True) -> list:
    """Return the list of ionizable resids from head3.lst.
    When argument 'as_string' is True, the output ia a 'ready-to-paste'
    string that can be pasted into the input parameter file to populate the
    argument 'correl_resids'.
    """
    h3_fp = Path(h3_fp)
    if h3_fp.name != "head3.lst":
        logger.error(f"File name not 'head3.lst': {h3_fp!s}")
        return []
    if not h3_fp.exists():
        logger.error(f"Not found: {h3_fp!s}")
        return []

    h3_lines = [line.split()[1] for line in h3_fp.read_text().splitlines()[1:]]
    h3_ioniz_res = list(
        dict((f"{res[:3]}{res[5:11]}", "") for res in h3_lines if res[:3] in IONIZABLES).keys()
    )
    if not as_string:
        return h3_ioniz_res

    res_lst = "[\n"
    for res in h3_ioniz_res:
        res_lst += f"{res!r},\n"

    return res_lst + "]"


def crgmsa_parser() -> ArgumentParser:

    DESC = """cms_analysis_wc.py ::
Stand-alone charge microstate analysis with correlation.
Can be used without the mcce program & its codebase.
The only command line option is the parameters file.
"""
    USAGE = """
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
"""
    p = ArgumentParser(
        prog="cms analysis with correlation",
        description=DESC,
        usage=USAGE,
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
            print(list_head3_ionizables(main_d.get("mcce_dir", ".") + "/head3.lst",
                                        as_string=True))
            sys.exit()

    # run the processing pipeline:
    crg_msa_with_correlation(main_d, crg_histo_d)
    return


if __name__ == "__main__":
    crgmsa_cli(sys.argv[1:])
