#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# unit and integration tests # TODO split both types of testing into sepeate files
import os
import sys
from pathlib import Path
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr

import unittest
from parameterized import parameterized  # equivalent to @pytest.mark.parametrize()

# import pytest

sys.path.append("../")  # search within parent dir of tests_folder
from attrici import estimator as est
import attrici.postprocess as pp
import attrici.sanity_check.estimation_quality_check as e
import settings as s


## get logger
# logger = s.init_logger("__test__")  # TODO replace print statement by logger message


## change variables, files and paths for test data
s.tile = "00003"
s.variable_hour = "18"
s.variable = "tas"

s.data_dir = "./test_data"
s.input_dir = Path(s.data_dir) / "input" / s.dataset / s.tile
s.output_dir = Path(s.data_dir) / "output" / s.dataset / s.tile

s.ts_dir = Path(f"./{s.output_dir}/timeseries/{s.variable}")
s.trace_dir = Path(f"./{s.output_dir}/traces/{s.variable}")
s.lsm_file = Path(s.input_dir) / f"landmask_{s.tile}_demo.nc"


class TestEstimator(unittest.TestCase):
    """
    test class Estimator
    """

    def setUp(self):
        file_dims = open(f"{s.input_dir}/dimensions.json", "r")
        self.dims = json.load(file_dims)["dimensions"]  # lat, lon, time
        self.dims["TIME0"] = datetime.now()  # replace time dimension by now
        file_dims.close()

        self.estimator = est.estimator(s)  ## call class to test

    # test data for parameterize test
    df_1, df_2 = [
        pd.read_csv(f"{s.input_dir}/{df_name}.csv", index_col="Unnamed: 0") for df_name in ["df_1", "df_2"]
    ]  # load input df for estimator()
    df_1["ds"], df_2["ds"] = [pd.to_datetime(date_col, format="%Y-%m-%d %H:%M:%S") for date_col in [df_1["ds"], df_2["ds"]]]  # use dates

    # class object to test, parametize unittest (equivalent to pytest's @pytest.mark.parametrize())
    @parameterized.expand([(21.258333, df_1), (21.308333333333398, df_2)])
    def test_estimate_parameters(self, sp_lon, df):  # speed up test by testing only gridpoints for latitude 62.3916
        # instance to test
        free_params = self.estimator.estimate_parameters(df, self.dims["sp_lat"], sp_lon, s.map_estimate, self.dims["TIME0"])[0]

        ## load reference data
        for trace_file in s.trace_dir.glob(f"lat_{self.dims['sp_lat']}/lon{sp_lon}*"):
            with open(trace_file, "rb") as trace:
                ref_free_params = pickle.load(trace)

        ## assert test data == reference data, check if free parameters equal element-wise within a tolerance
        for free_param, ref_free_param in zip(free_params.items(), ref_free_params.items()):
            self.assertTrue(np.allclose(free_param[1], ref_free_param[1], equal_nan=True))


class TestPostprocess(unittest.TestCase):
    def test_rescale_aoi(self):
        """
        unit test for input variable of rescale_aoi()
        """
        coord_list = np.array([62.39166667, 62.375, 62.35833333])
        coord_list_shuffled = np.array([62.375, 62.35833333, 62.39166667])
        coord_list_negative = np.array([-62.39166667, -62.375, 62.35833333])

        assert pp.rescale_aoi(coord_list, 62.375) == [1]
        assert pp.rescale_aoi(coord_list_shuffled, 62.375) == [0]
        assert pp.rescale_aoi(coord_list_negative, -62.375) == [1]


class TestOutputRunEstimation(unittest.TestCase):
    """
    test class for temporary output from run_estimation.py() and write_netcdf.py()
    """

    def test_number_files_equals_number_landcells(self):
        """
        test if enough land-cells were processed by comparing number of files with number of land cells
        """
        lsm = xr.load_dataset(s.lsm_file)
        nbr_landcells = lsm["area_European_01min"].count().values.tolist()
        print(f"Tile: {s.tile}, Variable: {s.variable, s.variable_hour}: {nbr_landcells} Land cells in lsm")

        print("Searching in", s.ts_dir)
        # nbr_files = e.count_files_in_directory(self.trace_dir, ".*")
        nbr_files = e.count_files_in_directory(s.ts_dir, "h5")

        self.assertEqual(nbr_files, nbr_landcells, f"{nbr_files} timeseries files <-> {nbr_landcells} number of land cells")

    ## TODO do monkeypatching or similar to imitate landmask, trace and timeseries files
    def test_occurrence_empty_files(self):
        """
        test if empty temporary files were created
        """
        ## ckeck for empty trace files
        trace_files = s.ts_dir.rglob("lon*")

        self.assertTrue(all([os.stat(file).st_size != 0 for file in trace_files]), f"empty file(s) exist in {s.ts_dir}")

    # ## TODO do monkeypatching or similar
    def test_number_failing_cells(self):
        """
        test if processing of cells failed
        """
        ## check amount of failing cells
        failing_cells = s.ts_dir.parent.parent / "./failing_cells.log"
        # print(failing_cells)
        with open(failing_cells, "r") as f:
            nbr_failcells = sum(1 for _ in f)

        self.assertEqual(nbr_failcells, 0, (f"failing cells in tile: {s.tile}," f"variable: {s.variable, s.variable_hour}"))


if __name__ == "__main__":
    unittest.main()
    print("Run all tests")
