"""Install-ready StarryNite parameter, distribution, and classifier assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ASSET_ROOT = Path(__file__).with_name("assets")
_PRESET_ROOT = _ASSET_ROOT / "presets"
_MODEL_ROOT = _ASSET_ROOT / "models"


@dataclass(frozen=True, slots=True)
class BundledStarryNitePreset:
    """One upstream new-MATLAB preset exposed as an install-ready choice."""

    preset_id: str
    display_name: str
    description: str
    filename: str

    @property
    def parameter_file(self) -> Path:
        return _PRESET_ROOT / self.filename


@dataclass(frozen=True, slots=True)
class BundledStarryNiteModel:
    """One source MAT model and its validated inert runtime export."""

    display_name: str
    mat_filename: str
    runtime_filename: str
    source_sha256: str

    @property
    def mat_file(self) -> Path:
        return _MODEL_ROOT / self.mat_filename

    @property
    def runtime_file(self) -> Path:
        return _MODEL_ROOT / self.runtime_filename


@dataclass(frozen=True, slots=True)
class BundledLegacyModelSource:
    """An upstream model that needs conversion with a compatible old MATLAB."""

    display_name: str
    mat_filename: str
    source_sha256: str
    compatibility_note: str

    @property
    def mat_file(self) -> Path:
        return _MODEL_ROOT / self.mat_filename


_PRESETS = (
    BundledStarryNitePreset(
        "dispim_singleview",
        "diSPIM single-view (2019)",
        "Single-view diSPIM preset; the recommended general StarryNite starting point.",
        "dispim_singleview_param.txt",
    ),
    BundledStarryNitePreset(
        "dispim_deconvolved",
        "diSPIM deconvolved (2019)",
        "Deconvolved diSPIM preset using the 2019 tracking model.",
        "dispim_mipav_decon_param.txt",
    ),
    BundledStarryNitePreset(
        "dispim_gaussian_model",
        "diSPIM Gaussian light-sheet model",
        "Deconvolved diSPIM preset with the additional Gaussian light-sheet classifier.",
        "dispim_mipav_decon_param_dispim_model.txt",
    ),
    BundledStarryNitePreset(
        "isim_red_40x",
        "iSIM red 40x",
        "Upstream iSIM red-channel 40x preset.",
        "iSIM_red_40xs.txt",
    ),
    BundledStarryNitePreset(
        "spinning_disk_red_40x",
        "Spinning-disk red 40x",
        "Upstream spinning-disk red-channel 40x preset.",
        "SD_red_40x.txt",
    ),
    BundledStarryNitePreset(
        "spinning_disk_red_60x_si",
        "Spinning-disk red 60x SI",
        "Upstream spinning-disk red-channel 60x structured-illumination preset.",
        "SD_red_60xSI.txt",
    ),
)

_MODELS = (
    BundledStarryNiteModel(
        "StarryNite 2019 tracking model",
        "2019TrackingModelv2.mat",
        "2019TrackingModelv2.atpy-model",
        "3431637f95caa91e3839d808de426b80f9e76d15e0ac5b6cc44d1a04647eb0fc",
    ),
    BundledStarryNiteModel(
        "Gaussian light-sheet tracking model",
        "gaussianlatedispimmodel_withoptimizedfeatures_ignoringFPstillpoorFN.mat",
        "gaussianlatedispimmodel_withoptimizedfeatures_ignoringFPstillpoorFN.atpy-model",
        "452fc2e74ab6ff846b34a999a684cb14d1d85904f71fc3dbeb88a0620d9bf1e6",
    ),
)

_LEGACY_MODEL_SOURCES = (
    BundledLegacyModelSource(
        "Pre-2019 red-channel single model",
        "clean_red_singlemodel_red_normal.mat",
        "5fff30745b5a36eb81475341405645b5820e8967c90dcadc2f61c916a0a0e807",
        (
            "Uses MATLAB's retired NaiveBayes object. Export it with the packaged "
            "helper and an older MATLAB release before exact Python replay."
        ),
    ),
)

DEFAULT_BUNDLED_PRESET_ID = "dispim_singleview"


def bundled_asset_root() -> Path:
    """Return the package directory containing third-party StarryNite assets."""

    return _ASSET_ROOT


def legacy_model_export_script() -> Path:
    """Return the packaged MATLAB helper used for old-release model conversion."""

    return (
        Path(__file__).with_name("oracle")
        / "matlab"
        / "export_starrynite_classifier_numeric.m"
    )


def bundled_parameter_presets() -> tuple[BundledStarryNitePreset, ...]:
    """Return all install-ready upstream ``newmatlab`` parameter choices."""

    return _PRESETS


def bundled_parameter_preset(preset_id: str) -> BundledStarryNitePreset:
    """Resolve one bundled preset by stable identifier."""

    requested = str(preset_id)
    for preset in _PRESETS:
        if preset.preset_id == requested:
            return preset
    raise KeyError(f"Unknown bundled StarryNite preset {requested!r}")


def default_bundled_parameter_file() -> Path:
    """Return the recommended ready-to-use StarryNite parameter source."""

    return bundled_parameter_preset(DEFAULT_BUNDLED_PRESET_ID).parameter_file


def is_bundled_parameter_file(path: str | Path) -> bool:
    """Return whether ``path`` is one of the package's immutable presets."""

    candidate = Path(path).resolve(strict=False)
    return any(
        preset.parameter_file.resolve(strict=False) == candidate
        for preset in _PRESETS
    )


def bundled_tracking_models() -> tuple[BundledStarryNiteModel, ...]:
    """Return source-bound models shipped ready for Python execution."""

    return _MODELS


def bundled_legacy_model_sources() -> tuple[BundledLegacyModelSource, ...]:
    """Return upstream model sources that deliberately fail closed until exported."""

    return _LEGACY_MODEL_SOURCES


def bundled_classifier_for_profile(profile: Any) -> Path | None:
    """Return a ready runtime model when the profile's MAT hash matches exactly."""

    digest = getattr(profile, "model_sha256", None)
    if not isinstance(digest, str):
        return None
    for model in _MODELS:
        if digest == model.source_sha256 and model.runtime_file.is_file():
            return model.runtime_file.resolve(strict=False)
    return None


def validate_bundled_assets() -> tuple[str, ...]:
    """Return missing asset descriptions; an empty tuple means the bundle is intact."""

    missing: list[str] = []
    for preset in _PRESETS:
        if not preset.parameter_file.is_file():
            missing.append(f"parameter preset {preset.filename}")
    for model in _MODELS:
        if not model.mat_file.is_file():
            missing.append(f"source model {model.mat_filename}")
        if not model.runtime_file.is_file():
            missing.append(f"runtime model {model.runtime_filename}")
    for model in _LEGACY_MODEL_SOURCES:
        if not model.mat_file.is_file():
            missing.append(f"legacy source model {model.mat_filename}")
    for filename in (
        "clean_distributions_newimage.mat",
        "clean_distributions_newimage_10thround.mat",
        "clean_distributions_newimage_10thround_edited2.mat",
    ):
        if not (_ASSET_ROOT / "distributions" / filename).is_file():
            missing.append(f"detector distribution {filename}")
    if not (_ASSET_ROOT / "LICENSE.GPL-3.0.txt").is_file():
        missing.append("StarryNite GPL-3.0 license")
    if not legacy_model_export_script().is_file():
        missing.append("legacy MATLAB model export script")
    return tuple(missing)


__all__ = [
    "DEFAULT_BUNDLED_PRESET_ID",
    "BundledStarryNiteModel",
    "BundledStarryNitePreset",
    "BundledLegacyModelSource",
    "bundled_asset_root",
    "bundled_classifier_for_profile",
    "bundled_legacy_model_sources",
    "bundled_parameter_preset",
    "bundled_parameter_presets",
    "bundled_tracking_models",
    "default_bundled_parameter_file",
    "is_bundled_parameter_file",
    "legacy_model_export_script",
    "validate_bundled_assets",
]
