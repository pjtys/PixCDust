#
# Copyright (C) 2024 Centre National d'Etudes Spatiales (CNES)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Interface used by all Pixcdust Converters."""

import copy
import operator
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd


class Converter:
    """Abstract class parent of pixcdust converters.

    They convert from official SWOT Pixel Cloud Netcdf to the supported format.

    Attributes:
        path_in: List of path of files to convert.
        variables: Optionally only read these variables.
        area_of_interest: Optionally only read points in area_of_interest.
        conditions: Optionally pass conditions to filter variables.\
                    Example: {\
                    "sig0":{'operator': "ge", 'threshold': 20},\
                    "classification":{'operator': "ge", 'threshold': 3},\
                    }
    """

    def __init__(
        self,
        path_in: str | Iterable[str] | Path | Iterable[Path],
        variables: list[str] | None = None,
        area_of_interest: gpd.GeoDataFrame | None = None,
        conditions: dict[str, dict[str, str | float]] | None = None,
    ):
        """Basic initialisation of a pixcdust converter.

        They convert from official SWOT Pixel Cloud Netcdf to the supported format.

        Args:
            path_in: Path or list of path of file(s) to convert.
            variables: Optionally only read these variables.
            area_of_interest: Optionally only read points in area_of_interest.
            compute_wse:  toggle water surface elevation computation.
            conditions: Optionally pass conditions to filter variables.\
                    Example: {\
                    "sig0":{'operator': "ge", 'threshold': 20},\
                    "classification":{'operator': "ge", 'threshold': 3},\
                    }
        """
        if isinstance(path_in, str | Path):
            self.path_in = [str(path_in)]
        else:
            self.path_in = [str(p) for p in path_in]

        self.variables = copy.copy(variables)
        self.area_of_interest = area_of_interest
        self.conditions = conditions

    def database_from_nc(self, path_out: str | Path, mode: str = "w") -> None:
        """Convert the path_in files to path_out.
        Args:
            path_out: Output path of the convertion.
            mode: Writing mode of the output. Must be 'w'(write/append) or 'o'(overwrite).
        """
        raise NotImplementedError


class ConverterWSE(Converter):
    """Abstract class parent of pixcdust converters supporting water surface elevation computation.

    They convert from official SWOT Pixel Cloud Netcdf to the supported format.

    Attributes:
        path_in: List of path of files to convert.
        variables: Optionally only read these variables.
        area_of_interest: Optionally only read points in area_of_interest.
    """

    def database_from_nc(
        self, path_out: str | Path, mode: str = "w", compute_wse: bool = True
    ) -> None:
        """Convert the path_in files to path_out.
        Args:
            path_out: Output path of the convertion.
            mode: Writing mode of the outpout. Must be 'w'(write/append) or 'o'(overwrite).
            compute_wse:  toggle water surface elevation computation.
        """
        raise NotImplementedError

    def _append_wse_vars(self):
        """Need some vars to compute wse."""
        if self.variables is not None:
            for var in self._get_vars_wse_computation():
                if var not in self.variables:
                    self.variables.append(var)

    def _compute_wse(self, gdf):
        gdf[self._get_name_wse_var()] = (
            gdf[self._get_vars_wse_computation()[0]]
            - gdf[self._get_vars_wse_computation()[1]]
        )

    @staticmethod
    def _get_vars_wse_computation() -> list[str]:
        """Names of fields used to compute wse."""
        return ["height", "geoid"]

    @staticmethod
    def _get_name_wse_var() -> str:
        """Output name for wse."""
        return "wse"


@dataclass
class GeoLayerH3Projecter:
    """Class for adding H3 projections to databases

    Attributes:
        data: data getting projected
        resolution: Resolution

    """

    data: gpd.GeoDataFrame
    resolution: int

    def filter_variable(self, conditions: dict[str, dict[str, str | float]]) -> None:
        """filters from xarray dataset based 
        on operator and threshold on specific variables

        Args:
            conditions (dict): specifies the filters. \
                Example: {\
                    "sig0":{'operator': "ge", 'threshold': 20},\
                    "classification":{'operator': "ge", 'threshold': 3},\
                    }

        Raises:
            IOError: if variable provided in conditions are not\
                in self.data.columns
            ValueError: if 'operator' and 'to' keys are not\
                in conditions
            AttributeError: if operator is not the function name of\
                the operator module
        """
        _k_operator = "operator"
        _k_to = "threshold"
        # Test if conditions dict meets specifications
        print(conditions)
        for var, condition in conditions.items():
            if var not in self.data.columns:
                raise OSError(
                    f"dict conditions expected existing\
                        variables (in {self.data.columns}),\
                        received {var}"
                )
            for instructions in condition:
                if instructions not in [_k_operator, _k_to]:
                    raise ValueError(
                        f"dict conditions expected {_k_to} and {_k_operator}\
                        keys in dict {conditions},\
                        received {instructions}"
                    )
            op_name = condition[_k_operator]
            # Typing issue, improvement would be to use TypedDict for conditions
            assert isinstance(op_name, str)
            print(f"operator.{op_name}")
            ope = getattr(operator, op_name)
            self.data = self.data[
                ope(
                    self.data[var],
                    condition[_k_to],
                )
            ]

    def compute_h3_layer(self) -> None:
        """Project data to h3."""
        from pixcdust.dggs import h3_tools

        self.data = h3_tools.gdf_to_h3_gdf(
            self.data,
            self.resolution,
        )

    def compute_healpix_layer(self) -> None:
        """Project data to Healpix."""
        from pixcdust.dggs import h3_tools

        self.data = h3_tools.gdf_to_healpix_gdf(
            self.data,
            self.resolution,
        )
