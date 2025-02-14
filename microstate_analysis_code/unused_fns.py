#!/usr/bin/env python

"""
Module: unused_fns.py

  Unused functions pulled out of ms_analysis_wc.py.
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import List


Kcal2kT = 1.688


def groupms_byenergy(microstates: list, ticks: List[float]) -> list:
    """
    Group the microstates' energies into bands provided in `ticks`.
    Args:
      microstates (list): List of microstates
      ticks (list(float)): List of energies; will be sorted.
    """
    N = len(ticks)
    ticks.sort()
    ticks.append(1.0e100)  # add a big number as the last boundary
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


def groupms_byconfid(microstates: list, confids: list, conformers: list) -> tuple:
    """
    Divide the microstates by the conformers ids provided in `confids`
    into 2 groups: the first contains ALL of the given conformers, the
    second one contains does not.
    Note: An ID is a match if it is a substring of the conformer name.
    Args:
      microstates (list): List of microstates
      confids (list): List of conformer ids.
    Return:
      A 2-tuple: Microstates with all of `confids`, microstates with some or none.
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


def groupms_byiconf(microstates: list, iconfs: list) -> tuple:
    """
    Divide the microstates by the conformers indices provided in `iconfs`
    into 2 groups: the first contains one of the given conformers, the
    second one contains none of the listed conformers.
    Args:
      microstates (list): List of microstates
      iconfs (list): List of conformer indices.
    Return:
      A 2-tuple: Microstates with any of `iconfs`, microstates with none
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


def ms_energy_stat(microstates: list) -> tuple:
    """
    Return the lowest, average, and highest energies of the listed
    microstates.
    """
    ms = next(iter(microstates))
    lowest_E = highest_E = ms.E
    N_ms = 0
    total_E = 0.0
    for ms in microstates:
        if lowest_E > ms.E:
            lowest_E = ms.E
        elif highest_E < ms.E:
            highest_E = ms.E
        N_ms += ms.count
        total_E += ms.E * ms.count

    average_E = total_E / N_ms

    return lowest_E, average_E, highest_E


def e2occ(energies: list) -> float:
    """Given a list of energy values in unit Kacl/mol,
    calculate the occupancy by Boltzmann Distribution.
    """
    e = np.array(energies)
    e = e - min(e)
    Pi_raw = np.exp(-Kcal2kT * e)

    return Pi_raw / sum(Pi_raw)


def bhata_distance(prob1: list, prob2: list) -> float:
    """Bhattacharyya distance between 2 probability distributions."""
    d_max = 10000.0  # Max possible value
    p1 = np.array(prob1) / sum(prob1)
    p2 = np.array(prob2) / sum(prob2)
    if len(p1) != len(p2):
        d = d_max
    else:
        bc = sum(np.sqrt(p1 * p2))
        if bc <= np.exp(-d_max):
            d = d_max
        else:
            d = -np.log(bc)
    return d


def whatchanged_conf(msgroup1: list, msgroup2: list) -> dict:
    """Given two groups of microstates, populate a dict with what
    changed at conformer level.
    """
    occ1 = ms_convert2occ(msgroup1)
    occ2 = ms_convert2occ(msgroup2)

    all_keys = list(set(occ1.keys()) | set(occ2.key()))
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


def whatchanged_res(msgroup1: list, msgroup2: list, free_res: list) -> list:
    """Given two groups of microstates, return a list of Bhattacharyya
    distances of free residues.
    """
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


def ms_convert2occ(microstates: list) -> dict:
    """
    Given a list of microstates, convert to conformer occupancy
    for conformers that appear at least once in the microstates.
    Return:
      A dict: {iconf: occ}
    """
    # TODO: use Counter
    occurence = {}  # dict of conformer occurence
    occ = {}
    N_ms = 0
    for ms in microstates:
        N_ms += ms.count
        for ic in ms.state:
            if ic in occurence:
                occurence[ic] += ms.count
            else:
                occurence[ic] = ms.count

    for key in occurence:
        occ[key] = occurence[key] / N_ms

    return occ


def jointplot(
    charge_ms_files: list,
    background_charge: float,
    fig_title: str,
    out_dir: str,
    save_name: str,
    show: bool = False,
):
    """Output a joint plot with histogram."""
    # Compute charge state population including background charge
    x_av = [sum(x) + background_charge for x in charge_ms_files[0]]

    # Avoid log(0) by replacing zeros with NaN (or a small positive value)
    y_av = [math.log10(x) if x > 0 else float("nan") for x in charge_ms_files[1]]

    # Initialize the JointGrid
    g1 = sns.JointGrid(marginal_ticks=True, height=6)

    # Scatter plot with bubble sizes based on energy differences
    ax = sns.scatterplot(
        x=x_av,
        y=y_av,
        size=charge_ms_files[3],
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

    if fig_title:
        g1.fig.suptitle(fig_title, fontsize=16)

    fig_fp = Path(out_dir).joinpath(save_name)
    # Save the figure
    g1.savefig(fig_fp, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {fig_fp!s}")
    if fig_fp.suffix != ".png":
        fig_png = fig_fp.with_suffix(".png")
        g1.savefig(fig_png, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {fig_png}")
    if show:
        plt.show()

    return

