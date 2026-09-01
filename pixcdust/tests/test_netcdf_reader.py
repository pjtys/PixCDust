"""Unit tests for :mod:`pixcdust.readers.netcdf`.

The tests are split in two groups:

* Pure/light tests that do not touch the filesystem (dataclasses, ``__init__``
  wiring, ``filter_variable`` logic, method dispatch). They build an in-memory
  dataset with :func:`pixcdust.tests.mock.mock_xarray`.
* File-based tests that read the bundled mock NetCDF
  (``mock_data/swot_pixc.nc``, exposed through the ``sample_nc`` /
  ``input_files`` fixtures from ``conftest.py``). They exercise
  ``extract_info_from_nc_attrs``, ``open_dataset`` and ``open_mfdataset``.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from pixcdust.readers.netcdf import (
    NcFormatCfg,
    NcSimpleConstants,
    NcSimpleReader,
)
from pixcdust.tests.mock import mock_xarray

# ---------------------------------------------------------------------------
# Known content of mock_data/swot_pixc.nc (see conftest.SAMPLE_NC)
# ---------------------------------------------------------------------------
EXPECTED_TILE = 163
EXPECTED_PASS = 33
EXPECTED_CYCLE = 15
EXPECTED_SWATH = "R"
EXPECTED_TIME_START_PREFIX = "2024-05-09T11:58:17"
EXPECTED_NB_POINTS = 10001


# ---------------------------------------------------------------------------
# Local fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def reader_with_mock() -> NcSimpleReader:
    """A reader whose ``data`` is forced to an in-memory dataset.

    ``__init__`` never opens the path, so a dummy path is enough here.
    """
    reader = NcSimpleReader("dummy_path.nc")
    reader.data = mock_xarray(length=200)
    return reader


# ---------------------------------------------------------------------------
# NcSimpleConstants
# ---------------------------------------------------------------------------
class TestNcSimpleConstants:
    def test_default_values(self):
        cst = NcSimpleConstants()
        assert cst.default_dim_name == "points"
        assert cst.default_long_name == "longitude"
        assert cst.default_lat_name == "latitude"
        assert cst.default_cyc_num_name == "cycle_number"
        assert cst.default_pass_num_name == "pass_number"
        assert cst.default_tile_num_name == "tile_number"
        assert cst.default_swath_side_name == "swath_side"
        assert cst.default_time_start_name == "time_granule_start"

    def test_added_names(self):
        """The 'added' names are used across converters/readers."""
        cst = NcSimpleConstants()
        assert cst.default_added_time_name == "time"
        assert cst.default_added_points_name == "points"

    def test_time_formats(self):
        cst = NcSimpleConstants()
        # attrs format is the one used to parse the granule start global attribute
        assert cst.default_time_format_attrs == "%Y-%m-%dT%H:%M:%S.%fZ"
        # filename format must render a granule timestamp as expected
        dt = datetime(2024, 5, 9, 11, 58, 17, tzinfo=UTC)
        assert dt.strftime(cst.default_time_format_filename) == "20240509T115817"


# ---------------------------------------------------------------------------
# NcFormatCfg
# ---------------------------------------------------------------------------
class TestNcFormatCfg:
    def test_defaults(self):
        cfg = NcFormatCfg()
        assert cfg.trusted_group == "pixel_cloud"
        assert isinstance(cfg.constants, NcSimpleConstants)
        assert "interferogram" in cfg.forbidden_variables
        assert "pixc_line_qual" in cfg.forbidden_variables

    def test_forbidden_variables_not_shared_between_instances(self):
        """default_factory must give each instance its own list."""
        cfg_a = NcFormatCfg()
        cfg_b = NcFormatCfg()
        assert cfg_a.forbidden_variables is not cfg_b.forbidden_variables
        cfg_a.forbidden_variables.append("mutated")
        assert "mutated" not in cfg_b.forbidden_variables


# ---------------------------------------------------------------------------
# __init__ wiring
# ---------------------------------------------------------------------------
class TestInit:
    def test_multi_file_support_flag(self):
        assert NcSimpleReader.MULTI_FILE_SUPPORT is True

    def test_single_path(self):
        reader = NcSimpleReader("a.nc")
        assert reader.multi_file_db is False
        assert reader.path == "a.nc"
        assert reader.trusted_group == "pixel_cloud"
        assert isinstance(reader.cst, NcSimpleConstants)
        assert reader.conditions is None
        assert "interferogram" in reader.forbidden_variables

    def test_list_path_enables_multi_file(self):
        reader = NcSimpleReader(["b.nc", "a.nc"])
        assert reader.multi_file_db is True
        assert isinstance(reader.path, list)
        assert all(isinstance(p, str) for p in reader.path)

    def test_conditions_stored(self):
        conditions = {"sig0": {"operator": "ge", "threshold": 20}}
        reader = NcSimpleReader("a.nc", conditions=conditions)
        assert reader.conditions == conditions

    def test_custom_format_cfg_is_honored(self):
        cfg = NcFormatCfg(trusted_group="custom_group", forbidden_variables=["foo"])
        reader = NcSimpleReader("a.nc", format_cfg=cfg)
        assert reader.trusted_group == "custom_group"
        assert reader.forbidden_variables == ["foo"]


# ---------------------------------------------------------------------------
# filter_variable (operates on self.data / self.conditions, no file needed)
# ---------------------------------------------------------------------------
class TestFilterVariable:
    def test_single_condition(self, reader_with_mock):
        reader_with_mock.conditions = {"sig0": {"operator": "ge", "threshold": 15}}
        reader_with_mock.filter_variable()
        assert bool((reader_with_mock.data["sig0"] >= 15).all())

    def test_multiple_conditions(self, reader_with_mock):
        reader_with_mock.conditions = {
            "sig0": {"operator": "gt", "threshold": 10},
            "height": {"operator": "le", "threshold": 5},
        }
        reader_with_mock.filter_variable()
        data = reader_with_mock.data
        assert bool((data["sig0"] > 10).all())
        assert bool((data["height"] <= 5).all())

    def test_none_conditions_is_noop(self, reader_with_mock):
        before = reader_with_mock.data["sig0"].size
        reader_with_mock.conditions = None
        reader_with_mock.filter_variable()
        assert reader_with_mock.data["sig0"].size == before

    def test_unknown_variable_raises_oserror(self, reader_with_mock):
        reader_with_mock.conditions = {"not_a_var": {"operator": "ge", "threshold": 1}}
        with pytest.raises(OSError):
            reader_with_mock.filter_variable()

    def test_missing_operator_key_raises_valueerror(self, reader_with_mock):
        reader_with_mock.conditions = {"sig0": {"threshold": 1}}
        with pytest.raises(ValueError):
            reader_with_mock.filter_variable()

    def test_missing_threshold_key_raises_valueerror(self, reader_with_mock):
        reader_with_mock.conditions = {"sig0": {"operator": "ge"}}
        with pytest.raises(ValueError):
            reader_with_mock.filter_variable()

    def test_invalid_operator_raises_attributeerror(self, reader_with_mock):
        reader_with_mock.conditions = {
            "sig0": {"operator": "not_an_operator", "threshold": 1}
        }
        with pytest.raises(AttributeError):
            reader_with_mock.filter_variable()


# ---------------------------------------------------------------------------
# read() dispatch and to_h3 / to_healpix variable selection (monkeypatched)
# ---------------------------------------------------------------------------
class TestReadDispatch:
    def test_read_single_file_calls_open_dataset(self, monkeypatch):
        reader = NcSimpleReader("a.nc")
        calls = []
        monkeypatch.setattr(reader, "open_dataset", lambda: calls.append("single"))
        monkeypatch.setattr(
            reader, "open_mfdataset", lambda orbit_info=False: calls.append("multi")
        )
        reader.read()
        assert calls == ["single"]

    def test_read_multi_file_calls_open_mfdataset(self, monkeypatch):
        reader = NcSimpleReader(["a.nc", "b.nc"])
        received = {}
        monkeypatch.setattr(
            reader, "open_dataset", lambda: received.setdefault("single", True)
        )
        monkeypatch.setattr(
            reader,
            "open_mfdataset",
            lambda orbit_info=False: received.update(orbit_info=orbit_info),
        )
        reader.read(orbit_info=True)
        assert received == {"orbit_info": True}


# ---------------------------------------------------------------------------
# extract_info_from_nc_attrs (uses the bundled mock NetCDF)
# ---------------------------------------------------------------------------
class TestExtractInfoFromNcAttrs:
    def test_returns_expected_orbit_info(self, sample_nc):
        (
            time_start,
            _dt_time_start,
            cycle_number,
            pass_number,
            tile_number,
            swath_side,
        ) = NcSimpleReader.extract_info_from_nc_attrs(str(sample_nc))

        assert time_start.startswith(EXPECTED_TIME_START_PREFIX)
        assert int(cycle_number) == EXPECTED_CYCLE
        assert int(pass_number) == EXPECTED_PASS
        assert int(tile_number) == EXPECTED_TILE
        assert swath_side == EXPECTED_SWATH

    def test_datetime_is_utc_aware_without_microseconds(self, sample_nc):
        _, dt_time_start, *_ = NcSimpleReader.extract_info_from_nc_attrs(str(sample_nc))
        assert isinstance(dt_time_start, datetime)
        assert dt_time_start.tzinfo is not None
        assert dt_time_start.utcoffset().total_seconds() == 0
        assert dt_time_start.microsecond == 0
        assert (dt_time_start.year, dt_time_start.month, dt_time_start.day) == (
            2024,
            5,
            9,
        )

    def test_numeric_fields_are_uint16(self, sample_nc):
        _, _, cycle_number, pass_number, tile_number, _ = (
            NcSimpleReader.extract_info_from_nc_attrs(str(sample_nc))
        )
        assert cycle_number.dtype == np.uint16
        assert pass_number.dtype == np.uint16
        assert tile_number.dtype == np.uint16


# ---------------------------------------------------------------------------
# open_dataset (single file, end-to-end on the mock NetCDF)
# ---------------------------------------------------------------------------
class TestOpenDataset:
    def test_loads_pixel_cloud_group(self, sample_nc):
        reader = NcSimpleReader(str(sample_nc))
        reader.open_dataset()
        assert reader.data is not None
        assert "height" in reader.data
        assert reader.data.sizes["points"] == EXPECTED_NB_POINTS

    def test_postprocess_adds_points_coordinate(self, sample_nc):
        reader = NcSimpleReader(str(sample_nc))
        reader.open_dataset()
        assert reader.cst.default_added_points_name in reader.data.coords

    def test_variable_subset(self, sample_nc):
        reader = NcSimpleReader(str(sample_nc), variables=["height", "sig0"])
        reader.open_dataset()
        assert set(reader.data.data_vars) == {"height", "sig0"}

    def test_conditions_are_applied(self, sample_nc):
        reader = NcSimpleReader(
            str(sample_nc),
            variables=["height", "sig0"],
            conditions={"sig0": {"operator": "ge", "threshold": 30}},
        )
        reader.open_dataset()
        assert bool((reader.data["sig0"] >= 30).all())
        # filtering should have removed at least one point
        assert reader.data.sizes["points"] < EXPECTED_NB_POINTS

    def test_to_geodataframe_after_open(self, sample_nc):
        reader = NcSimpleReader(str(sample_nc), variables=["height"])
        reader.open_dataset()
        gdf = reader.to_geodataframe()
        assert "height" in gdf.columns
        assert gdf.geometry.notna().all()
        assert len(gdf) == EXPECTED_NB_POINTS


# ---------------------------------------------------------------------------
# open_mfdataset (multi-file path, exercised with the single mock file)
# ---------------------------------------------------------------------------
class TestOpenMfdataset:
    def test_with_orbit_info_appends_orbit_variables(self, input_files):
        reader = NcSimpleReader(
            input_files, variables=["height", "sig0", "classification"]
        )
        reader.open_mfdataset(orbit_info=True)
        cst = reader.cst
        for name in (
            cst.default_tile_num_name,
            cst.default_cyc_num_name,
            cst.default_pass_num_name,
            cst.default_added_time_name,
        ):
            assert name in reader.data
        assert reader.data.sizes["points"] == EXPECTED_NB_POINTS

    def test_postprocess_adds_points_coordinate(self, input_files):
        reader = NcSimpleReader(input_files, variables=["height"])
        reader.open_mfdataset(orbit_info=False)
        assert reader.cst.default_added_points_name in reader.data.coords

    def test_requesting_forbidden_variable_raises(self, input_files):
        reader = NcSimpleReader(input_files, variables=["height", "interferogram"])
        with pytest.raises(OSError):
            reader.open_mfdataset(orbit_info=False)

    @pytest.mark.xfail(
        reason=(
            "open_mfdataset nests filter/postprocess under `if self.variables:`, "
            "so with variables=None the points coordinate is never added "
            "(inconsistent with open_dataset)."
        ),
        strict=False,
    )
    def test_postprocess_runs_even_without_variable_subset(self, input_files):
        reader = NcSimpleReader(input_files)  # variables=None
        reader.open_mfdataset(orbit_info=False)
        assert reader.cst.default_added_points_name in reader.data.coords


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
