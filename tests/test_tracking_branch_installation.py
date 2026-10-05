"""Regression checks for branch-safe feature installation."""

import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "alpha-v2"


@pytest.mark.parametrize("checkout_state", ["correct", "wrong", "detached"])
def test_native_installer_enforces_checkout_state(
    checkout_state: str,
    tmp_path: Path,
) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required to exercise the branch guard")

    repo = tmp_path / "checkout"
    initial_branch = EXPECTED_BRANCH if checkout_state != "wrong" else "wrong-branch"
    subprocess.run(
        [git, "init", "--initial-branch", initial_branch, str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )

    if checkout_state == "detached":
        (repo / "fixture.txt").write_text("fixture\n", encoding="utf-8")
        subprocess.run(
            [git, "-C", str(repo), "add", "fixture.txt"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                git,
                "-C",
                str(repo),
                "-c",
                "user.name=AceTree test",
                "-c",
                "user.email=acetree-test@example.invalid",
                "commit",
                "-m",
                "installer fixture",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [git, "-C", str(repo), "checkout", "--detach"],
            check=True,
            capture_output=True,
            text=True,
        )

    scripts_dir = repo / "scripts"
    scripts_dir.mkdir()
    if os.name == "nt":
        shell = shutil.which("powershell.exe") or shutil.which("pwsh")
        if shell is None:
            pytest.skip("PowerShell is required to exercise the Windows installer")
        script_name = "install_tracking_integration.ps1"
        command = [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(scripts_dir / script_name),
            "-DryRun",
        ]
    else:
        shell = shutil.which("sh")
        if shell is None:
            pytest.skip("A POSIX shell is required to exercise the Unix installer")
        script_name = "install_tracking_integration.sh"
        command = [shell, str(scripts_dir / script_name), "--dry-run"]

    shutil.copy2(REPO_ROOT / "scripts" / script_name, scripts_dir / script_name)
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if checkout_state == "correct":
        assert result.returncode == 0, output
        assert f"from branch '{EXPECTED_BRANCH}'" in output
    elif checkout_state == "wrong":
        assert result.returncode != 0
        assert "current branch is 'wrong-branch'" in output
    else:
        assert result.returncode != 0
        assert "detached checkout" in output.lower()


@pytest.mark.parametrize(
    "variant,version,system,machine,unsupported",
    [
        ("gui", (3, 9), "Linux", "x86_64", True),
        ("core", (3, 9), "Darwin", "arm64", True),
        ("gui", (3, 10), "Darwin", "x86_64", False),
        ("all", (3, 13), "Darwin", "x86_64", False),
        ("gui", (3, 14), "Darwin", "x86_64", True),
        ("all", (3, 14), "Darwin", "x86_64", True),
        ("core", (3, 14), "Darwin", "x86_64", False),
        ("gui", (3, 14), "Darwin", "arm64", False),
        ("gui", (3, 14), "Windows", "AMD64", False),
        ("gui", (3, 14), "Linux", "x86_64", False),
    ],
)
def test_installer_runtime_compatibility(variant, version, system, machine, unsupported):
    preflight = runpy.run_path(str(REPO_ROOT / "scripts" / "installer_preflight.py"))
    error = preflight["compatibility_error"](variant, version, system, machine)
    assert bool(error) is unsupported
    if unsupported and version >= (3, 14):
        assert "Intel macOS" in error
        assert "Python 3.10-3.13" in error
        assert "core" in error


def test_native_installer_stops_before_pip_on_preflight_failure(tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    if os.name == "nt":
        shell = shutil.which("powershell.exe") or shutil.which("pwsh")
        if shell is None:
            pytest.skip("PowerShell is required to exercise the Windows installer")
        script_name = "install_tracking_integration.ps1"
        command = [
            shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(scripts_dir / script_name), "-Python", sys.executable,
        ]
    else:
        shell = shutil.which("sh")
        if shell is None:
            pytest.skip("A POSIX shell is required to exercise the Unix installer")
        script_name = "install_tracking_integration.sh"
        command = [shell, str(scripts_dir / script_name), "--python", sys.executable]
    shutil.copy2(REPO_ROOT / "scripts" / script_name, scripts_dir / script_name)
    (scripts_dir / "installer_preflight.py").write_text(
        'raise SystemExit("preflight rejection fixture")\n', encoding="utf-8"
    )
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "preflight rejection fixture" in output
    assert "Obtaining" not in output
