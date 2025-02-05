#!/usr/bin/env python

"""
Module: Microstates Analysis with Weighted Correlation on Charge Microstates

Last updated: 2025-02-05
"""
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
from pathlib import Path
import seaborn as sns
import sys
from scipy.stats import rankdata
from typing import Union


ph2Kcal = 1.364
Kcal2kT = 1.688


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


ACIDIC_RES = ["ASP", "GLU"]
BASIC_RES = ["ARG", "HIS", "LYS"]
POLAR_RES = ["CYS", "TYR"]


class Microstate:
    def __init__(self, state, E, count):
        self.state = state
        self.E = E
        self.count = count


class Conformer:
    def __init__(self):
        self.iconf = 0
        self.ires = 0
        self.confid = ""
        self.resid = ""
        self.occ = 0.0
        self.crg = 0.0

    def load_from_head3lst(self, line):
        fields = line.split()
        self.iconf = int(fields[0]) - 1
        self.confid = fields[1]
        self.resid = self.confid[:3] + self.confid[5:11]
        self.crg = float(fields[4])


def reader_gen(fpath: Path):
    """
    Generator function yielding a file line.
    """
    with open(fpath) as fh:
        for line in fh:
            yield line


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
                    print("This file %s is not a valid microstate file" % fname)
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
            print("The microstates dict is empty.")
            return []

        kind = kind.lower()
        if kind not in ["deterministic", "random"]:
            raise ValueError(
                f"Values for `kind` are 'deterministic' or 'random'; Given: {kind}"
            )

        ms_sampled = []
        ms_list = list(self.microstates.values())
        counts = ms_counts(ms_list)  # total number of ms
        sampled_cumsum = np.cumsum([mc.count for mc in ms_list])

        if kind == "deterministic":
            sampled_ms_indices = np.arange(
                size, counts - size, counts / size, dtype=int
            )
        else:
            rng = np.random.default_rng(seed=seed)
            sampled_ms_indices = rng.integers(
                low=0, high=counts, size=size, endpoint=True
            )

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
            print(
                "'sort_by' must be a valid microstate attribute; choices: ['count', 'E']"
            )
            return None

        return sorted(
            list(self.microstates.values()),
            key=operator.attrgetter(sort_by),
            reverse=sort_reverse,
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
        Usage:
          ```
          # Define df, then get the weighted correlation:
          ```
          wcorr = WeightedCorr(xyw=df[["xcol", "ycol", "wcol"]])(method='pearson')
          ```
        """
        if (df is None) and (wcol is None):
            if np.all([i is None for i in [xyw, x, y, w]]):
                raise ValueError("No data supplied")

            if not (
                (isinstance(xyw, pd.DataFrame))
                != (np.all([isinstance(i, pd.Series) for i in [x, y, w]]))
            ):
                raise TypeError(
                    "xyw should be a pd.DataFrame, or x, y, w should be pd.Series"
                )

            xyw = pd.concat([x, y, w], axis=1).dropna() if xyw is None else xyw.dropna()
            self.x, self.y, self.w = (
                pd.to_numeric(xyw[i], errors="coerce").values for i in xyw.columns
            )
            self.df = None

        elif (wcol is not None) and (df is not None):
            if (not isinstance(df, pd.DataFrame)) or (not isinstance(wcol, str)):
                raise ValueError(
                    "df should be a pd.DataFrame and wcol should be a string"
                )

            if wcol not in df.columns:
                raise KeyError("wcol not found in column names of df")

            self.df = df.loc[:, [x for x in df.columns if x != wcol]]
            self.w = pd.to_numeric(df.loc[:, wcol], errors="coerce")

        else:
            raise ValueError(
                "Incorrect arguments specified, please specify xyw, or (x, y, w) or (df, wcol)"
            )

    def _wcov(self, x, y, ms):
        return np.sum(self.w * (x - ms[0]) * (y - ms[1]))

    def _pearson(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        mx, my = (np.sum(i * self.w) / np.sum(self.w) for i in [x, y])

        return self._wcov(x, y, [mx, my]) / np.sqrt(
            self._wcov(x, x, [mx, mx]) * self._wcov(y, y, [my, my])
        )

    def _wrank(self, x):
        (unique, arr_inv, counts) = np.unique(
            rankdata(x), return_counts=True, return_inverse=True
        )
        a = np.bincount(arr_inv, self.w)
        return (np.cumsum(a) - a)[arr_inv] + ((counts + 1) / 2 * (a / counts))[arr_inv]

    def _spearman(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        return self._pearson(self._wrank(x), self._wrank(y))

    def __call__(self, method: str = "pearson") -> Union[float,]:
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
            out = pd.DataFrame(np.nan, index=self.df.columns, columns=self.df.columns)
            for i, x in enumerate(self.df.columns):
                for j, y in enumerate(self.df.columns):
                    if i >= j:
                        out.loc[x, y] = cor(
                            x=pd.to_numeric(self.df[x], errors="coerce"),
                            y=pd.to_numeric(self.df[y], errors="coerce"),
                        )
                        out.loc[y, x] = out.loc[x, y]
            return out


def get_free_res_ids(conformers: list, free_residues: list) -> list:
    return [conformers[confs[0]].resid for confs in free_residues]


def ms2crgms(ll: list, dd: dict) -> list:
    """
    Given a list with format: [[ms.E, ms.count, ms.state], ] and sorted by E,
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


def iconf2crg(conformers: list) -> dict:
    """Map mcce conformers indices to their charges.
    {conf.iconf:conf.crg for conf in msa.conformers}
    """
    return {conf.iconf: conf.crg for conf in conformers}


def groupms_byenergy(microstates, ticks):
    """
    This function takes in a list of microstates and a list of energy
    numbers (N values), divide the microstates into N bands by using
    the energy number as lower boundaries. The list of energy will be
    sorted ascendingly.
    """
    N = len(ticks)
    ticks.sort()
    ticks.append(1.0e100)  # add a big number as the rightest-most boundary
    resulted_bands = [[] for i in range(N)]

    for ms in microstates:
        it = -1
        for itick in range(N):
            if ticks[itick] <= ms.E < ticks[itick + 1]:
                it = itick
                break
        if it >= 0:
            resulted_bands[it].append(ms)

    return resulted_bands


def groupms_byiconf(microstates, iconfs):
    """
    This function takes in a list of microstates and a list of conformer
    indicies, divide microstates into two groups: the first one is those
    contain one of the given conformers, the second one is those contain
    none of the listed conformers.
    """
    ingroup = []
    outgroup = []
    for ms in microstates:
        contain = False
        for ic in iconfs:
            if ic in ms.state:
                ingroup.append(ms)
                contain = True
                break
        if not contain:
            outgroup.append(ms)

    return ingroup, outgroup


def groupms_byconfid(microstates, confids):
    """
    Group conformers by conformer IDs. IDs are in a list and ID is
    considered as a match as long as it is a substring of the conformer
    name. The selected microstates must have all conformers and returned
    in the first group, and the rest are in the second group.
    """
    ingroup = []
    outgroup = []
    for ms in microstates:
        contain = True
        names = [conformers[ic].confid for ic in ms.state]
        for confid in confids:
            innames = False
            for name in names:
                if confid in name:
                    innames = True
                    break
            contain = contain and innames
        if contain:
            ingroup.append(ms)
        else:
            outgroup.append(ms)

    return ingroup, outgroup


def ms_energy_stat(microstates):
    """
    Given a list of microstates, find the lowest energy, average energy, and highest energy
    """
    ms = next(iter(microstates))
    lowerst_E = highest_E = ms.E
    N_ms = 0
    total_E = 0.0
    for ms in microstates:
        if lowerst_E > ms.E:
            lowerst_E = ms.E
        elif highest_E < ms.E:
            highest_E = ms.E
        N_ms += ms.count
        total_E += ms.E * ms.count

    average_E = total_E / N_ms

    return lowerst_E, average_E, highest_E


def ms_convert2occ(microstates):
    """
    Given a list of microstates, convert to conformer occupancy
    of conformers appeared at least once in the microstates.
    """
    occurance = {}  # occurance of conformer, as a dictionary
    occ = {}
    N_ms = 0
    for ms in microstates:
        N_ms += ms.count
        for ic in ms.state:
            if ic in occurance:
                occurance[ic] += ms.count
            else:
                occurance[ic] = ms.count

    for key in occurance.keys():
        occ[key] = occurance[key] / N_ms

    return occ


def ms_counts(microstates):
    """
    Calculate total counts of microstates
    """
    N_ms = 0
    for ms in microstates:
        N_ms += ms.count

    return N_ms


def ms_charge(ms):
    """Compute microstate charge"""
    crg = 0.0
    for ic in ms.state:
        crg += conformers[ic].crg
    return crg


def ms_convert2sumcrg(microstates, free_res):
    """
    Given a list of microstates, convert to net charge of each free residue.
    """
    iconf2ires = {}
    for i_res in range(len(free_res)):
        for iconf in free_res[i_res]:
            iconf2ires[iconf] = i_res

    charges_total = [0.0 for i in range(len(free_res))]
    N_ms = 0
    for ms in microstates:
        N_ms += ms.count
        for ic in ms.state:
            ir = iconf2ires[ic]
            charges_total[ir] += conformers[ic].crg * ms.count

    charges = [x / N_ms for x in charges_total]

    return charges


def read_conformers(head3_path: str = "head3.lst") -> list:
    """Load conformerrs from given head3.lst path;
    Uses ./head3.lst by default; returns empty list if file not found.
    """
    conformers = []
    if not Path(head3_path).exists():
        return conformers

    with open(head3_path) as h3:
        lines = h3.readlines()
    lines.pop(0)
    for line in lines:
        conf = Conformer()
        conf.load_from_head3lst(line)
        conformers.append(conf)

    return conformers


def e2occ(energies):
    """Given a list of energy values in Kcal/mol units, calculate the occupancy
    by Boltzmann Distribution."""
    e = np.array(energies)
    e = e - min(e)
    Pi_raw = np.exp(-Kcal2kT * e)
    return Pi_raw / sum(Pi_raw)


def bhata_distance(prob1, prob2):
    d_max = 10000.0  # Max possible value set to this
    p1 = np.array((prob1)) / sum(prob1)
    p2 = np.array((prob2)) / sum(prob2)
    if len(p1) != len(p2):
        d = d_max
    else:
        bc = sum(np.sqrt(p1 * p2))
        # print(bc, np.exp(-d_max))
        if bc <= np.exp(-d_max):
            d = d_max
        else:
            d = -np.log(bc)

    return d


def whatchanged_conf(msgroup1, msgroup2):
    """Given two group of microstates, calculate what changed at conformer level."""
    occ1 = ms_convert2occ(msgroup1)
    occ2 = ms_convert2occ(msgroup2)

    all_keys = set(occ1.keys())
    all_keys |= set(occ2.keys())
    # FIX: clearer: all_keys = set(occ1.keys()) | set(occ2.keys())

    all_keys = list(all_keys)
    all_keys.sort()
    diff_occ = {}
    for key in all_keys:
        if key in occ1:
            p1 = occ1[key]
        else:
            p1 = 0.0
        if key in occ2:
            p2 = occ2[key]
        else:
            p2 = 0.0
        diff_occ[key] = p2 - p1

    return diff_occ


def whatchanged_res(msgroup1, msgroup2, free_res):
    """Return a list of Bhatachaya Distance of free residues."""
    occ1 = ms_convert2occ(msgroup1)
    occ2 = ms_convert2occ(msgroup2)

    bhd = []
    for res in free_res:
        p1 = []
        p2 = []
        for ic in res:
            if ic in occ1:
                p1.append(occ1[ic])
            else:
                p1.append(0.0)
            if ic in occ2:
                p2.append(occ2[ic])
            else:
                p2.append(0.0)
        bhd.append(bhata_distance(p1, p2))

    return bhd


# populate conformers list, if possible:
conformers = read_conformers()


def renameOrderResidues(crgms_data: pd.DataFrame):
    rename_dict = {}
    acid_list = []
    base_list = []
    polar_list = []
    ub_q_list = []
    non_res_list = []

    # exclude Occupancy column:
    for col in crgms_data.columns[:-1]:
        residue_number = col[4:8]

        if residue_number.isdigit():  # Check if the substring is numeric
            # remove leading zeros
            # rename_dict[col] = f"{col[3]}_{col[:3]}{int(residue_number)}"
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
            resout = "MQ8" + y[5:]
            ub_q_list.append(resout)
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


def choose_res_data(df: pd.DataFrame, choose_res: list):
    """Group the df by the given list."""
    if not choose_res:
        raise TypeError("Error: empty list given for df.groupby 'by' argument.")

    df_choose_res = df.groupby(choose_res).Count.sum().reset_index()
    df_res_sort = df_choose_res.sort_values(by="Count", ascending=False).reset_index(
        drop=True
    )

    return df_res_sort


def findUniqueCrgmsCountOrder(
    crg_list_ms: list, begin_energy: float = None, end_energy: float = None
):
    """
    Args:
      crg_list_ms (list): List of charge microstate, assumed sorted.
      begin_energy (float, None): Lower energy bound.
      end_energy (float, None): Upper energy bound.

    Returns:
     A tuple of lists: all_crg_ms_unique, all_count, unique_crg_state_order, energy_diff_all

    Notes:
        1. To filter crg_list_ms based on energy, both energy bounds are needed.
        2. Variable unique_crg_state_order gives the order of unique charge state based on energy.
           Lowest energy charge state will give the order 1; the second will give the order 2.
    """
    if not begin_energy and not end_energy:
        print("No energy bounds given: All microstates are selected.")
        begin_energy = crg_list_ms[0][0]
        end_energy = crg_list_ms[-1][0]

        print(f"Number of charge ms: {len(crg_list_ms):,}")

    elif begin_energy and end_energy:
        # both energy bounds given; filter the input list energy accordingly:
        crg_list_ms = [
            [x[0], x[1], x[2]]
            for x in crg_list_ms
            if x[0] >= begin_energy and x[0] <= end_energy
        ]
        print(f"Number of filtered charge ms: {len(crg_list_ms):,}")
    else:
        sys.exit("Both energy bounds are needed.")

    # unique charge as key and energy, count and order
    crg_all_count = {}
    unique_crg_state_order = 1
    for array in crg_list_ms:
        if tuple(array[2]) not in crg_all_count:
            crg_all_count[(tuple(array[2]))] = [
                array[1],
                [array[0]],
                [unique_crg_state_order],
            ]
            unique_crg_state_order += 1
        else:
            crg_all_count[(tuple(array[2]))][0] += array[1]

            # get the maximum and minimum energy
            min_energy = min(min(crg_all_count[(tuple(array[2]))][1]), array[0])
            max_energy = max(max(crg_all_count[(tuple(array[2]))][1]), array[0])

            # update the energy list with the minimum and maximum energy
            crg_all_count[(tuple(array[2]))][1].clear()
            crg_all_count[(tuple(array[2]))][1].append(min_energy)
            crg_all_count[(tuple(array[2]))][1].append(max_energy)

    # make a list of count, unique charge microstate, energy difference and order.
    all_crg_ms_unique = []
    all_count = []
    energy_diff_all = []
    unique_crg_state_order = []
    for u, v in crg_all_count.items():
        all_crg_ms_unique.append(list(u))
        all_count.append(v[0])
        unique_crg_state_order.append(v[2][0])
        if len(v[1]) == 2:
            energy_diff_all.append(round(v[1][1] - v[1][0], 6))
        elif len(v[1]) == 1:
            energy_diff_all.append(0.0)
        else:
            sys.exit(
                "Error while creating unique charge state: len(v[1]) is neither 1 or 2."
            )

    print(f"Number of unique charge ms: {len(all_crg_ms_unique):,}")

    return all_crg_ms_unique, all_count, unique_crg_state_order, energy_diff_all


def ConcaCrgMsPandas(
    unique_crg_ms_list,
    ms_count,
    ms_order,
    free_residues,
    background_charge,
    residue_interest_list,
) -> pd.DataFrame:
    """Process inputs into a single pandas.DataFrame."""
    unique_crg_ms_list_pd = pd.DataFrame(unique_crg_ms_list).T
    ms_count_pd = pd.DataFrame(ms_count, columns=["Count"]).T
    ms_order_pd = pd.DataFrame(ms_order, columns=["Order"]).T
    crg_ms_count_pd = pd.concat([unique_crg_ms_list_pd, ms_count_pd, ms_order_pd])
    crg_count_res_1 = pd.concat([free_residues, crg_ms_count_pd], axis=1)
    crg_count_res_1.loc["Count", "Residue"] = "Count"
    crg_count_res_1.loc["Order", "Residue"] = "Order"
    all_crg_count_res = crg_count_res_1.set_index("Residue")

    # sort based on the count
    all_crg_count_res = all_crg_count_res.sort_values(
        by="Count", axis=1, ascending=False
    )
    all_crg_count_res.columns = range(all_crg_count_res.shape[1])
    all_crg_count_res = all_crg_count_res.T.set_index("Order")
    all_crg_count_res["Occupancy"] = round(
        all_crg_count_res["Count"] / sum(all_crg_count_res["Count"]), 3
    )
    all_crg_count_res["Sum_crg_protein"] = (
        all_crg_count_res.iloc[:, :-2].sum(axis=1) + background_charge
    )
    crg_count_res = all_crg_count_res.copy()
    for i in all_crg_count_res.columns:
        if (
            i[:3] not in residue_interest_list
            and i != "Occupancy"
            and i != "Count"
            and i != "Sum_crg_protein"
        ):
            crg_count_res.drop([i], axis=1, inplace=True)

    return crg_count_res


def jointplot(
    charge_ms_files: list,
    background_charge: float,
    out_dir: str,
    fig_name: str,
    show: bool = False,
):
    """Output a joint plot with histogram."""
    # Compute charge state population including background charge
    x_av = [sum(x) + background_charge for x in charge_ms_files[0]]

    # Avoid log(0) by replacing zeros with NaN (or a small positive value)
    y_av = [math.log10(x) if x > 0 else float("nan") for x in charge_ms_files[1]]

    # Convert energy differences to float
    # energy_diff_all_fl = [float(x) for x in charge_ms_files[3]]

    # Initialize the JointGrid
    g1 = sns.JointGrid(marginal_ticks=True, height=6)

    # Scatter plot with bubble sizes based on energy differences
    ax = sns.scatterplot(
        x=x_av,
        y=y_av,
        size=charge_ms_files[3],
        # size=[float(x) for x in charge_ms_files[3]],
        sizes=(10, 500),
        ax=g1.ax_joint,
    )

    # Set x-axis ticks based on the range of x_av
    ax.set_xticks(range(int(min(x_av)), int(max(x_av)) + 1))

    # Labels
    ax.set_xlabel("Charge", fontsize=15)
    ax.set_ylabel(r"log$_{10}$(Count)", fontsize=16)

    # Adjust legend position
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # Histogram on the marginal x-axis
    ax2 = sns.histplot(x=x_av, linewidth=2, discrete=True, ax=g1.ax_marg_x)
    ax2.set_ylabel(None)

    # Remove y-axis marginal plot
    g1.ax_marg_y.set_axis_off()

    # Adjust layout and title
    g1.fig.subplots_adjust(top=0.9)

    # FIX: suptitle
    g1.fig.suptitle("All Microstates Energy (dry2_semi)_pH7", fontsize=16)

    mc_energy_fig = Path(out_dir).joinpath(fig_name)
    # Save the figure
    g1.savefig(mc_energy_fig, dpi=600, bbox_inches="tight")
    print(f"Figure saved: {mc_energy_fig!s}")
    if show:
        plt.show()

    return


if __name__ == "__main__":
    msout = MSout("ms_out/pH7eH0ms.txt")
    # e_step = (msout.highest_E - msout.lowest_E)/20
    # ticks = [msout.lowest_E + e_step*(i) for i in range(20)]
    # ms_in_bands = groupms_byenergy(msout.microstates.values(), ticks)
    # print([len(band) for band in ms_in_bands])
    #     netural, charged = groupms_byiconf(msout.microstates.values(), [12, 13, 14, 15])
    #     l_E, a_E, h_E = ms_energy_stat(msout.microstates.values())
    #     print(l_E, a_E, h_E)

    # charge over energy bands
    # e_step = (msout.highest_E - msout.lowest_E) / 20
    # ticks = [msout.lowest_E + e_step*(i+1) for i in range(19)]
    # ms_in_bands = groupms_byenergy(msout.microstates.values(), ticks)
    # for band in ms_in_bands:
    #     band_total_crg = 0.0
    #     for ms in band:
    #         band_total_crg += ms_charge(ms)
    #     print(band_total_crg/ms_counts(band))

    # netural, charged = groupms_byiconf(msout.microstates.values(), [12, 13, 14, 15])
    # diff_occ = whatchanged_conf(netural, charged)
    # for key in diff_occ.keys():
    #     print("%3d, %s: %6.3f" % (key, conformers[key].confid, diff_occ[key]))

    # diff_bhd = whatchanged_res(netural, charged, msout.free_residues)
    # for ir in range(len(msout.free_residues)):
    #     print("%s: %6.4f" % (conformers[msout.free_residues[ir][0]].resid, diff_bhd[ir]))
    # charges = ms_convert2sumcrg(msout.microstates.values(), msout.free_residues)
    # for ir in range(len(msout.free_residues)):
    #     print("%s: %6.4f" % (conformers[msout.free_residues[ir][0]].resid, charges[ir]))
    microstates = list(msout.microstates.values())
    glu35_charged, _ = groupms_byconfid(microstates, ["GLU-1A0035"])
    print(len(microstates))
    print(len(glu35_charged))
