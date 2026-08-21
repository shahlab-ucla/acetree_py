"""Regression checks for branch-safe feature installation."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "subcellular-measurements"
EXPECTED_CLONE = (
    "git clone --branch subcellular-measurements --single-branch "
    "https://github.com/shahlab-ucla/acetree_py.git"
)


@pytest.mark.parametrize("document", ["README.md", "docs/user_guide.md"])
def test_install_docs_pin_feature_branch(document: str) -> None:
    text = (REPO_ROOT / document).read_text(encoding="utf-8")

    assert EXPECTED_CLONE in text
    assert "python -m acetree_py --version" in text


@pytest.mark.parametrize(
    "script",
    [
        "scripts/install_tracking_integration.ps1",
        "scripts/install_tracking_integration.sh",
    ],
)
def test_installers_fail_closed_on_wrong_named_branch(script: str) -> None:
    text = (REPO_ROOT / script).read_text(encoding="utf-8")

    assert EXPECTED_BRANCH in text
    assert "branch --show-current" in text
    assert "current branch" in text.lower()
    assert "--editable" in text
    assert "tracking integration" in text


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


def test_architecture_install_notes_branch_requirement() -> None:
    text = (REPO_ROOT / "docs/architecture.md").read_text(encoding="utf-8")

    assert "branch-pinned clone" in text
    assert "../README.md#installation" in text


def test_plugin_plan_keeps_tracking_as_its_upstream() -> None:
    text = (REPO_ROOT / "PLUGIN_MIGRATION_PLAN.md").read_text(encoding="utf-8")

    assert "off `tracking-integration`" in text
    assert "off `main`" not in text
