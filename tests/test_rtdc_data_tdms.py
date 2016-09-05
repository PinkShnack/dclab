#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Downsampling only affects RTDC_DataSet._plot_filter
"""
from __future__ import print_function

import codecs
import numpy as np
import os
from os.path import abspath, dirname, join
import shutil
import sys
import tempfile
import warnings
import zipfile


# Add parent directory to beginning of path variable
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from dclab import RTDC_DataSet

from helper_methods import example_data_dict, retreive_tdms, example_data_sets


def test_load_tdms_simple():
    tdms_path = retreive_tdms(example_data_sets[0])
    ds = RTDC_DataSet(tdms_path)
    assert ds._filter.shape[0] == 156

def test_load_tdms_all():
    for ds in example_data_sets:
        tdms_path = retreive_tdms(ds)
        ds = RTDC_DataSet(tdms_path)

def test_load_tdms_avi_files():
    tdms_path = retreive_tdms(example_data_sets[1])
    edest = dirname(tdms_path)
    ds1 = RTDC_DataSet(tdms_path)
    assert ds1.video == "M1_imaq.avi"
    shutil.copyfile(join(edest, "M1_imaq.avi"),
                    join(edest, "M1_imag.avi"))
    ds2 = RTDC_DataSet(tdms_path)
    # prefer imag over imaq
    assert ds2.video == "M1_imag.avi"
    shutil.copyfile(join(edest, "M1_imaq.avi"),
                    join(edest, "M1_test.avi"))
    ds3 = RTDC_DataSet(tdms_path)
    # ignore any other videos
    assert ds3.video == "M1_imag.avi"
    os.remove(join(edest, "M1_imaq.avi"))
    os.remove(join(edest, "M1_imag.avi"))
    ds4 = RTDC_DataSet(tdms_path)
    # use available video if ima* not there
    assert ds4.video == "M1_test.avi"

            

if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    