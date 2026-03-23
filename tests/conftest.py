"""Shared test fixtures for AceTree-Py tests."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── Sample nucleus text lines (new format) ──
# Columns: index, status, pred, succ1, succ2, x, y, z, size, identity, weight,
#           rweight, rsum, rcount, assignedID, rwraw, rwcorr1, rwcorr2, rwcorr3, rwcorr4

SAMPLE_NUC_LINE_1 = "1, 1, -1, 1, -1, 300, 250, 15.0, 20, ABa, 5000, 100, 50, 25, , 120, 10, 15, 12, 8"
SAMPLE_NUC_LINE_2 = "2, 1, -1, 2, 3, 400, 300, 16.5, 22, P1, 6000, 200, 60, 30, , 250, 20, 25, 18, 12"
SAMPLE_NUC_LINE_DEAD = "3, -1, -1, -1, -1, 100, 100, 5.0, 10, , 1000, 0, 0, 0, , 0, 0, 0, 0, 0"
SAMPLE_NUC_LINE_ASSIGNED = "4, 1, 1, 4, -1, 350, 280, 14.0, 18, ABa, 4500, 90, 45, 20, ForcedName, 110, 8, 12, 10, 6"
SAMPLE_NUC_LINE_NILL = "5, 1, nill, nill, -1, 200, 200, 10.0, 15, P0, 3000, 50, 20, 10, , 60, 5, 8, 6, 4"

# Minimal line with fewer columns (no red data)
SAMPLE_NUC_LINE_MINIMAL = "1, 1, -1, 1, -1, 300, 250, 15.0, 20, ABa, 5000"


@pytest.fixture
def sample_nuclei_zip(tmp_path: Path) -> Path:
    """Create a minimal nuclei ZIP archive for testing.

    Contains 3 timepoints with a few nuclei each.
    """
    zip_path = tmp_path / "test_nuclei.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Timepoint 1: 2 nuclei (P0 before first division)
        zf.writestr(
            "nuclei/t001-nuclei",
            "1, 1, -1, 1, -1, 300, 250, 15.0, 20, P0, 5000, 100, 50, 25, , 120, 10, 15, 12, 8\n"
        )

        # Timepoint 2: 2 nuclei (AB and P1 after first division)
        zf.writestr(
            "nuclei/t002-nuclei",
            "1, 1, 1, 1, -1, 280, 240, 14.0, 18, AB, 4800, 90, 45, 20, , 110, 8, 12, 10, 6\n"
            "2, 1, 1, 2, -1, 320, 260, 16.0, 22, P1, 5200, 110, 55, 28, , 130, 12, 18, 14, 10\n"
        )

        # Timepoint 3: 3 nuclei (ABa, ABp, P1 continuing)
        zf.writestr(
            "nuclei/t003-nuclei",
            "1, 1, 1, -1, -1, 260, 230, 13.0, 16, ABa, 4600, 85, 42, 18, , 105, 7, 10, 8, 5\n"
            "2, 1, 1, -1, -1, 300, 250, 15.0, 20, ABp, 4500, 80, 40, 17, , 100, 6, 9, 7, 4\n"
            "3, 1, 2, -1, -1, 340, 270, 17.0, 24, P1, 5100, 105, 52, 26, , 125, 11, 16, 13, 9\n"
        )

    return zip_path


@pytest.fixture
def sample_xml_config(tmp_path: Path, sample_nuclei_zip: Path) -> Path:
    """Create a minimal XML config file for testing."""
    config_path = tmp_path / "test_config.xml"
    config_path.write_text(
        f"""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="{sample_nuclei_zip}"/>
    <image file="test_images/img_t001-p01.tif"/>
    <end index="350"/>
    <naming method="NEWCANONICAL"/>
    <resolution xyRes="0.09" zRes="1.0" planeEnd="30"/>
    <exprCorr type="blot"/>
</embryo>
"""
    )
    return config_path


@pytest.fixture
def sample_auxinfo_v1(tmp_path: Path) -> Path:
    """Create a sample AuxInfo v1 CSV file."""
    path = tmp_path / "testAuxInfo.csv"
    path.write_text(
        "name,slope,intercept,xc,yc,maj,min,ang,zc,zslope,time,zpixres,axis\n"
        "test,0.9,-27,360,255,585,390,0,14,10.4,160,11.1,ADL\n"
    )
    return path


@pytest.fixture
def sample_auxinfo_v2(tmp_path: Path) -> Path:
    """Create a sample AuxInfo v2 CSV file."""
    path = tmp_path / "testAuxInfo_v2.csv"
    path.write_text(
        "name,slope,intercept,xc,yc,maj,min,zc,zslope,time,zpixres,AP_orientation,LR_orientation\n"
        "test,0.9,-27,360,255,585,390,14,10.4,160,11.1,-1 0 0,0 0 1\n"
    )
    return path
