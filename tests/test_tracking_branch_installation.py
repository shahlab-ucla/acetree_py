"""Regression checks for branch-safe feature installation."""

import os
from pathlib import Path
import shutil
import subprocess

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


