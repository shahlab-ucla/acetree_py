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
if os.environ.get('ACETREE_SOFTWARE_OPENGL') == '1':
    from ctypes.util import find_library
    library = find_library('opengl32')
    assert library and Path(library).resolve() == Path(os.environ['QT_OPENGL_DLL']).resolve(), library
    print('Windows PyOpenGL library:', library, flush=True)
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
    import vispy
    print(vispy.sys_info(), flush=True)
finally:
    viewer.close()
"""


WINDOWS_OPENGL_DLL = """
from pathlib import Path
import importlib
import qtpy
binding = importlib.import_module(qtpy.API_NAME)
root = Path(binding.__file__).resolve().parent
libraries = sorted(root.rglob('opengl32sw.dll'))
assert len(libraries) == 1, f'Expected one bundled Qt Mesa DLL under {root}: {libraries}'
library = libraries[0]
print('Windows Qt binding:', qtpy.API_NAME, 'Mesa DLL:', library, flush=True)
Path('windows-opengl-dll.txt').write_text(str(library), encoding='utf-8')
"""

LINUX_QT_LIBRARIES = """
from pathlib import Path
import importlib
import os
import subprocess
import qtpy
binding = importlib.import_module(qtpy.API_NAME)
root = Path(binding.__file__).resolve().parent
plugins = sorted(root.glob('Qt*/plugins/platforms/libqxcb.so'))
plugins += sorted(root.glob('Qt*/plugins/xcbglintegrations/libqxcb-*.so'))
print('Qt binding:', qtpy.API_NAME, 'root:', root, flush=True)
for name in ('DISPLAY', 'QT_QPA_PLATFORM', 'LIBGL_ALWAYS_SOFTWARE'):
    print(name, '=', os.environ.get(name), flush=True)
assert plugins, 'Could not locate installed Qt xcb platform plugin'
missing = []
for plugin in plugins:
    print('ldd:', plugin, flush=True)
    result = subprocess.run(['ldd', str(plugin)], capture_output=True, text=True, timeout=10)
    print(result.stdout, result.stderr, flush=True)
    if result.returncode or 'not found' in result.stdout:
        missing.append(str(plugin))
assert not missing, 'Missing Qt native dependencies: ' + ', '.join(missing)
"""

QT_PLATFORM_DIAGNOSTICS = """
import os
os.environ['QT_DEBUG_PLUGINS'] = '1'
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.processEvents()
print('Qt platform initialized:', app.platformName(), flush=True)
"""

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--suite', choices=('full', 'smoke'), default='smoke')
    parser.add_argument('--qt', choices=('default', 'pyside6'), default='default')
    parser.add_argument('--software-opengl', action='store_true',
                        help='Use the installed Qt Mesa DLL on Windows CI virtual displays')
    parser.add_argument('--expected-architecture', choices=('x64', 'arm64'))
    args = parser.parse_args()
    if args.software_opengl and os.name != 'nt':
        parser.error('--software-opengl is supported only on Windows')
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
                       ACETREE_EXPECTED_QT=args.qt, ACETREE_SOFTWARE_OPENGL='0',
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
            try:
                result = subprocess.run(command, cwd=cwd, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                if check:
                    raise
                result = subprocess.CompletedProcess(command, 124)
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
        if args.software_opengl:
            run('gui-windows-opengl-library', [str(gui), '-I', '-c', WINDOWS_OPENGL_DLL],
                timeout=30)
            library = (workspace / 'windows-opengl-dll.txt').read_text(encoding='utf-8')
            if not Path(library).is_file():
                raise RuntimeError(f'Installed Qt Mesa DLL does not exist: {library}')
            # PyOpenGL searches PATH for opengl32.dll. Qt and both VisPy backends
            # must use this same Mesa DLL, including napari 0.7's gl+ backend.
            alias = gui.parent / 'opengl32.dll'
            shutil.copy2(library, alias)
            environment.update(QT_OPENGL='software', QT_OPENGL_DLL=str(alias),
                               VISPY_GL_LIB=str(alias), ACETREE_SOFTWARE_OPENGL='1',
                               PATH=str(gui.parent) + os.pathsep + environment.get('PATH', ''))
            print('Windows software OpenGL: Qt, VisPy and PyOpenGL use', alias, flush=True)
        if sys.platform.startswith('linux'):
            status = run('gui-linux-qt-libraries', [str(gui), '-I', '-c', LINUX_QT_LIBRARIES],
                         timeout=40, check=False)
            if status:
                run('gui-qt-platform-diagnostics',
                    [str(gui), '-I', '-c', QT_PLATFORM_DIAGNOSTICS], timeout=30, check=False)
                raise RuntimeError('Qt native dependency validation failed; see diagnostic logs')
        try:
            run('gui-imports-opengl', [str(gui), '-I', '-c', GUI_SMOKE])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            run('gui-qt-platform-diagnostics',
                [str(gui), '-I', '-c', QT_PLATFORM_DIAGNOSTICS], timeout=30, check=False)
            raise
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
