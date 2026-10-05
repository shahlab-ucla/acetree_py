"""Validate a wheel in fresh core and GUI environments, outside the checkout.

Run with the Python version being tested, for example:
    python scripts/validate_python_compatibility.py --workspace /tmp/acetree-ci --suite full
On Linux, run under Xvfb with Mesa; Windows and macOS use their native displays.
The workspace must be new or empty. It retains logs, versions, JUnit and screenshots.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import venv
import xml.etree.ElementTree as ET
from pathlib import Path

REQUIRED_GUI_TESTS = (
    "test_compact_workspace_preserves_selection_actions_menus_and_windows",
    "test_roi_job_is_responsive_and_publishes_only_current_results",
    "test_nuclear_job_stages_privately_and_commits_only_current_outputs",
    "test_application_quit_drains_owned_worker_and_discards_unpublished_result",
    "test_tracking_preview_round_trips_real_napari_2d_and_3d",
)

CORE_SMOKE = """
from pathlib import Path
import importlib.util
import sys
import numpy, scipy, skimage, tifffile, matplotlib, typer
import acetree_py
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.__main__ import app
assert Path(acetree_py.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert importlib.util.find_spec('napari') is None, 'Core install unexpectedly includes GUI'
assert importlib.util.find_spec('qtpy') is None, 'Core install unexpectedly includes Qt'
assert NucleiManager().num_timepoints == 0
print('Installed core:', acetree_py.__file__, flush=True)
"""

GUI_SMOKE = """
from pathlib import Path
import os
import sys
import numpy as np
import napari
import qtpy
from qtpy.QtWidgets import QApplication
from acetree_py.gui.app import AceTreeApp
import acetree_py
expected = os.environ['ACETREE_EXPECTED_QT']
assert (qtpy.API_NAME == 'PySide6' if expected == 'pyside6'
        else qtpy.API_NAME in ('PyQt5', 'PyQt6')), qtpy.API_NAME
assert Path(acetree_py.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print('Installed GUI:', acetree_py.__file__, 'napari', napari.__version__,
      'Qt', qtpy.API_NAME, qtpy.QT_VERSION, flush=True)
viewer = napari.Viewer(show=True)
try:
    viewer.add_image(np.arange(256 * 256, dtype=np.float32).reshape(256, 256))
    QApplication.processEvents()
    pixels = viewer.screenshot(path='opengl-smoke.png', canvas_only=True)
    assert pixels.ndim == 3 and min(pixels.shape[:2]) > 1, pixels.shape
    assert np.ptp(pixels[..., :3]) > 0, 'OpenGL canvas produced a blank image'
    print('OpenGL canvas:', pixels.shape, flush=True)
finally:
    viewer.close()
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--suite', choices=('full', 'smoke'), default='smoke')
    parser.add_argument('--qt', choices=('default', 'pyside6'), default='default')
    parser.add_argument('--expected-architecture', choices=('x64', 'arm64'))
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve()
    if workspace.is_relative_to(source):
        parser.error('--workspace must be outside the checkout')
    if workspace.exists() and any(workspace.iterdir()):
        parser.error('--workspace must be new or empty (fresh environments are required)')
    workspace.mkdir(parents=True, exist_ok=True)
    machine = platform.machine().lower()
    expected = {'x64': {'amd64', 'x86_64'}, 'arm64': {'arm64', 'aarch64'}}
    if args.expected_architecture and machine not in expected[args.expected_architecture]:
        raise RuntimeError(f'Expected {args.expected_architecture}, got {machine}')
    if sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise RuntimeError('This matrix supports standard GIL Python, not free-threaded Python')
    (workspace / 'runtime.txt').write_text(
        f'{sys.version}\n{platform.platform()}\n{machine}\n{sys.executable}\n',
        encoding='utf-8',
    )
    environment = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'QT_API', 'PYTEST_QT_API',
                'QT_QPA_PLATFORM', 'PYTEST_ADDOPTS'):
        environment.pop(key, None)
    environment.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
                       ACETREE_EXPECTED_QT=args.qt,
                       NUMBA_CACHE_DIR=str(workspace / 'numba-cache'))
    if args.qt == 'pyside6':
        environment.update(QT_API='pyside6', PYTEST_QT_API='pyside6')
    if sys.platform.startswith('linux'):
        if not environment.get('DISPLAY'):
            raise RuntimeError('Linux GUI validation requires Xvfb or a real X11 display')
        environment['QT_QPA_PLATFORM'] = 'xcb'
    interpreters: dict[str, Path] = {}

    def run(label: str, command: list[str], *, cwd: Path = workspace,
            timeout: int = 900, check: bool = True) -> int:
        print(f'[{label}] {command}', flush=True)
        with (workspace / f'{label}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=cwd, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
        print((workspace / f'{label}.log').read_text(encoding='utf-8', errors='replace'),
              flush=True)
        if check and result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        return result.returncode

    def create(name: str) -> Path:
        directory = workspace / f'{name}-env'
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(directory)
        executable = directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        interpreters[name] = executable
        run(f'{name}-pip-upgrade', [str(executable), '-m', 'pip', 'install', '--upgrade', 'pip'])
        return executable

    try:
        build = create('build')
        run('build-tools', [str(build), '-m', 'pip', 'install', 'build'])
        run('build-wheel', [str(build), '-m', 'build', '--wheel', '--outdir',
                            str(workspace / 'dist'), str(source)])
        wheels = list((workspace / 'dist').glob('*.whl'))
        if len(wheels) != 1:
            raise RuntimeError(f'Expected one built wheel, got {wheels}')
        wheel = str(wheels[0])
        core = create('core')
        run('core-install', [str(core), '-m', 'pip', 'install', wheel])
        run('core-pip-check', [str(core), '-m', 'pip', 'check'])
        run('core-imports', [str(core), '-I', '-c', CORE_SMOKE])
        for flag in ('--version', '--help'):
            run(f'core-cli-{flag[2:]}', [str(core), '-I', '-m', 'acetree_py', flag])
        gui = create('gui')
        if args.qt == 'pyside6':
            napari_range = '>=0.7.1,<0.8' if sys.version_info >= (3, 14) else '>=0.5,<0.7'
            requirements = [wheel + '[dev]', f'napari[pyside6,optional]{napari_range}',
                            'qtpy>=2.3']
        else:
            requirements = [wheel + '[gui,dev]']
        run('gui-install', [str(gui), '-m', 'pip', 'install', *requirements])
        run('gui-pip-check', [str(gui), '-m', 'pip', 'check'])
        run('gui-imports-opengl', [str(gui), '-I', '-c', GUI_SMOKE])
        # Tests import helpers as tests.*; copy them without any application source.
        shutil.copytree(source / 'tests', workspace / 'tests',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(source / 'scripts', workspace / 'scripts',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(source / 'pyproject.toml', workspace / 'pyproject.toml')
        common = [str(gui), '-m', 'pytest', '-ra', '-m', 'not matlab_oracle',
                  '--basetemp', str(workspace / 'pytest-gui-tmp')]
        report = workspace / 'gui-required.xml'
        run('gui-required', [*common, 'tests/test_workflow_workspace.py',
                             'tests/test_measurement_jobs.py',
                             'tests/test_tracking_napari_smoke.py',
                             f'--junitxml={report}'], timeout=300)
        cases = list(ET.parse(report).iter('testcase'))
        if not cases or any(case.find('skipped') is not None for case in cases):
            raise RuntimeError('Required GUI tests must execute; skips cannot validate compatibility')
        for name in REQUIRED_GUI_TESTS:
            if not any(case.get('name', '').split('[')[0] == name for case in cases):
                raise RuntimeError(f'Required GUI test did not execute: {name}')
        if args.suite == 'full':
            run('full-suite', [*common, '--basetemp', str(workspace / 'pytest-full-tmp'),
                               'tests', f'--junitxml={workspace / "full-suite.xml"}'],
                timeout=1800)
    finally:
        for name, executable in interpreters.items():
            run(f'{name}-versions', [str(executable), '-m', 'pip', 'freeze', '--all'], check=False)
            run(f'{name}-pip-final-check', [str(executable), '-m', 'pip', 'check'], check=False)


if __name__ == '__main__':
    main()
