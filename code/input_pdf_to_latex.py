"""
Read in h5 files containing inputs of MCMAC and output a table for paper
"""

from __future__ import (division, print_function)
import sys
import os
import pandas as pd
# import h5py
sys.path.append(os.path.abspath("../"))
from plot_cred_int import round_to_n_sig_fig
# from astropy.stats import biweight_midvariance as S_BI
# from astropy.stats import biweight_location as C_BI


def compile_data_lines(df, df_keys, row_labels, n_sig_fig=4):
    return [
        "{0} & {1} & {2} ,  {3} & {4} , {5} \\\\".format(
            row_labels[i],
            round_to_n_sig_fig(df.ix[key, "loc"], n_sig_fig),
            round_to_n_sig_fig(df.ix[key, "low68"], n_sig_fig),
            round_to_n_sig_fig(df.ix[key, "up68"], n_sig_fig),
            round_to_n_sig_fig(df.ix[key, "low95"], n_sig_fig),
            round_to_n_sig_fig(df.ix[key, "up95"], n_sig_fig)
        ) for i, key in enumerate(df_keys)
    ]


def compile_line_list(df, df_keys, row_labels, table_properties):
    line_lists = [
        "\\begin{table}",
        "\caption{" + table_properties['title'] + "}",
        "\\begin{center}",
        "\\begin{tabular}{@{}lcccc}\n",
        "\hline \hline Offset (kpc) & Location & 68\% CI$^\\dagger$ & 95\% CI$^\\dagger$ \\\\ \hline\n",
    ] + compile_data_lines(df, df_keys, row_labels) + [
        "\hline",
        "\end{tabular} ",
        "\end{center} ",
        "\label{tab:" + "{0}".format(table_properties['shortcut_label']) + "}",
        "\\footnotesize{$\dagger$ CI stands for the credible interval centered " +
        "on the biweight location estimate, " +
        "with 68\% of the probability density contained in the 68\% credible interval.}\\\\",
        "\end{table}"
    ]

    line_lists = [line + "\n" for line in line_lists]

    return line_lists


if __name__ == "__main__":
    data_path = "../data/"
    h5_file = "test_sum_stat.h5"
    paper2 = "/Users/karenyng/Documents/Illustris_analysis/paper/"
    F = open(paper2 + "input_table.tex", "w")

    full_sample_tab_prop = {
        "title": "Offsets for the full sample of 43 clusters at 2 kpc resolution.",
        "shortcut_label": "full2kpc_offsets",
    }

    full_sample_abs_tab_prop = {
        "title":
        "Absolute offsets for the full sample of 43 clusters at 2 kpc resolution.",
        "shortcut_label": "full_2kpc_abs_offsets",
    }

    df = pd.read_hdf(data_path + h5_file, key="df")
    abs_df = pd.read_hdf(data_path + h5_file, key="abs_df")
    row_labels = [
        "BCG",
        "Weighted centroid",
        "Shrinking aperture",
        "KDE"
    ]

    df_keys = [
        "BCG",
        "centroid",
        "shrink_cent",
        "KDE",
    ]

    lines = compile_line_list(df, df_keys, row_labels, full_sample_tab_prop)
    abs_lines = compile_line_list(abs_df, df_keys, row_labels,
                                  full_sample_abs_tab_prop)
    map(print, lines)
    map(print, abs_lines)
    F.writelines(lines)
    F.writelines(abs_lines)
    F.close()
