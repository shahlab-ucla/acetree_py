# AceTree-Py: napari Plugin Migration Plan

**Status:** Planning
**Branch:** `plugin-variant` (off `main`)
**Goal:** Repackage AceTree-Py as a napari npe2 plugin so it can be activated from within an existing napari session, while preserving the standalone CLI entry point and all current functionality.

---

## 1. Branch Strategy

Create a long-lived branch `plugin-variant` off of `main`. All plugin migration work happens on this branch. `main` continues as the standalone application.

```
main (standalone app — acetree-py gui config.xml)
  └── plugin-variant (napari plugin — activated from within napari)
```

Periodic merges from `main → plugin-variant` keep the plugin branch current with bug fixes and features. The two branches share all non-GUI code identically; only `gui/app.py`, `pyproject.toml`, and a few new files diverge.

---

## 2. Scope

### In scope
- npe2 plugin manifest (`napari.yaml`)
- Refactor `AceTreeApp` to receive a `Viewer` instead of creating one
- Replace private napari API usage (`_qt_window`, `_dock_widgets`)
- Plugin entry point in `pyproject.toml`
- Config loading via a plugin dialog (replaces CLI `acetree-py gui config.xml`)
- Preserve standalone CLI mode (`acetree-py gui`) as a thin wrapper

### Out of scope (future work)
- Publishing to napari-hub or PyPI
- napari reader/writer contributions for nuclei ZIP files
- Sample data contributions
- Theme contributions

---

## 3. Current napari API Surface (Audit)

### Files that touch napari (3 of 10 GUI files):

| File | Interaction | API type |
|------|------------|----------|
| `gui/app.py` | Viewer creation, dock widgets, keyboard binding, image layers, event loop, save/error dialogs | Mixed public + **3 private** |
| `gui/viewer_integration.py` | Shapes layers, mouse callbacks, layer selection | All public |
| `gui/contrast_tools.py` | `layer.contrast_limits`, `layer.data` | All public |

### Private API usage (all in `gui/app.py`):

| Line(s) | Code | Purpose | Plugin replacement |
|---------|------|---------|-------------------|
| 261, 282 | `self.viewer.window._qt_window` | Parent widget for `QFileDialog` and `QMessageBox` | Use `napari.current_viewer().window._qt_window` or `None` as parent (Qt handles parentless dialogs fine) |
| 688 | `self.viewer.window._qt_window` | Access `menuBar()` to add Window menu actions | Replaced by npe2 `menus` contributions |
| 706 | `self.viewer.window._dock_widgets` | Iterate dock wrappers for `toggleViewAction()` | Replaced by npe2 widget contributions (napari manages toggle actions automatically for contributed widgets) |

### Files with ZERO napari imports (already plugin-ready):
- `gui/cell_info_panel.py`
- `gui/edit_panel.py`
- `gui/lineage_widget.py`
- `gui/lineage_list.py`
- `gui/lineage_layout.py`
- `gui/player_controls.py`
- All `core/`, `naming/`, `editing/`, `io/`, `analysis/` modules

---

## 4. Implementation Steps

### Step 1: Create branch and add npe2 manifest

**Files:** `napari.yaml` (new), `pyproject.toml` (modified)

Create `acetree_py/napari.yaml`:

```yaml
name: acetree-py
display_name: AceTree-Py
contributions:
  commands:
    - id: acetree-py.open_config
      title: Open AceTree Config...
      python_name: acetree_py.gui._plugin:open_config_dialog

    - id: acetree-py.main_widget
      title: AceTree Main
      python_name: acetree_py.gui._plugin:create_main_widget

    - id: acetree-py.new_lineage_panel
      title: New Lineage Panel...
      python_name: acetree_py.gui._plugin:create_lineage_panel

  widgets:
    - command: acetree-py.main_widget
      display_name: AceTree
      autogenerate: false

  menus:
    napari/file:
      - command: acetree-py.open_config
    napari/plugins:
      - command: acetree-py.main_widget
    napari/window:
      - command: acetree-py.new_lineage_panel

  keybindings:
    - command: acetree-py.save
      key: Ctrl+S
    - command: acetree-py.save_as
      key: Ctrl+Shift+S
    - command: acetree-py.undo
      key: Ctrl+Z
    - command: acetree-py.redo
      key: Ctrl+Y
```

Add to `pyproject.toml`:

```toml
[project.entry-points."napari.manifest"]
acetree-py = "acetree_py:napari.yaml"
```

**Functionality impact:** None. This only adds metadata; no existing code changes.

---

### Step 2: Create plugin bridge module

**Files:** `acetree_py/gui/_plugin.py` (new)

This module provides the factory functions referenced by `napari.yaml`. It bridges between napari's plugin system and `AceTreeApp`.

```python
"""napari plugin entry points for AceTree-Py."""
from __future__ import annotations

import napari

_app_instance: AceTreeApp | None = None

def _get_or_create_app(viewer: napari.Viewer) -> AceTreeApp:
    """Get the singleton AceTreeApp for this viewer, or create one."""
    global _app_instance
    if _app_instance is None or _app_instance.viewer is not viewer:
        from .app import AceTreeApp
        _app_instance = AceTreeApp(manager=..., viewer=viewer)
    return _app_instance

def open_config_dialog(viewer: napari.Viewer):
    """Show a file dialog to load an AceTree XML config."""
    # QFileDialog to pick config.xml
    # Load NucleiManager from config
    # Create AceTreeApp with the existing viewer
    # Dock all widgets
    ...

def create_main_widget(viewer: napari.Viewer) -> QWidget:
    """Factory for the main AceTree control widget (napari widget contribution)."""
    ...

def create_lineage_panel(viewer: napari.Viewer):
    """Factory for creating a new lineage panel via dialog."""
    ...
```

**Functionality impact:** None — new file, no changes to existing code.

---

### Step 3: Refactor `AceTreeApp.__init__` to accept an existing Viewer

**Files:** `gui/app.py` (modified)

Currently:
```python
def __init__(self, manager, image_provider=None):
    self.viewer = None  # Created later in launch()

def launch(self):
    self.viewer = napari.Viewer(title="AceTree")  # CREATES viewer
    # ... add dock widgets ...

def run(self):
    self.launch()
    napari.run()  # STARTS event loop
```

Change to:
```python
def __init__(self, manager, image_provider=None, viewer=None):
    self.viewer = viewer  # Accept existing viewer
    # ... rest unchanged ...

def launch(self):
    if self.viewer is None:
        import napari
        self.viewer = napari.Viewer(title="AceTree")  # Standalone mode
    # ... add dock widgets (same as before) ...

def run(self):
    self.launch()
    if not _is_event_loop_running():
        import napari
        napari.run()  # Only start loop if not already running
```

The helper `_is_event_loop_running()` checks `QApplication.instance().thread().isRunning()` to avoid calling `napari.run()` when the plugin is activated inside an already-running napari.

**Functionality impact:** The standalone `acetree-py gui config.xml` path is preserved identically — `viewer=None` triggers the old behavior (create viewer + start event loop). The plugin path passes an existing viewer and skips `napari.run()`.

**Verification:** The `AceTreeApp.from_config()` classmethod and all `__main__.py` CLI commands remain untouched and continue to work as before.

---

### Step 4: Eliminate `_qt_window` usage for dialogs

**Files:** `gui/app.py` (modified)

Currently (3 occurrences):
```python
# Line 261 — save dialog
QFileDialog.getSaveFileName(self.viewer.window._qt_window, ...)

# Line 282 — error message
QMessageBox.critical(self.viewer.window._qt_window, ...)

# Line 688 — menu bar access
qt_window = self.viewer.window._qt_window
menu_bar = qt_window.menuBar()
```

Replace dialog parents with `None` (Qt creates a top-level dialog):
```python
# Line 261
QFileDialog.getSaveFileName(None, "Save Nuclei As", ...)

# Line 282
QMessageBox.critical(None, "Save Failed", ...)
```

Replace menu bar access with npe2 menu contributions (Step 1 already declares the menus in `napari.yaml`), so the entire `_add_panel_menu_actions()` method and its `_qt_window` + `_dock_widgets` access can be deleted.

**Functionality impact:** Dialogs behave identically — `QFileDialog` and `QMessageBox` with `parent=None` are standard Qt practice and appear as top-level windows centered on the screen instead of centered on the napari window. This is a negligible visual difference. The Window menu toggle actions for dock widgets are handled natively by napari for contributed widgets, so the manual `toggleViewAction()` logic is no longer needed.

**Risk assessment:** Low. Parentless Qt dialogs are well-supported. The only behavioral difference is window centering (screen center vs napari center).

---

### Step 5: Eliminate `_dock_widgets` usage

**Files:** `gui/app.py` (modified)

The entire `_add_panel_menu_actions()` method (lines 688–712) exists solely to:
1. Add toggle actions for dock widgets to the Window menu
2. Add a "New Lineage Panel..." action

Both are replaced by npe2 contributions:
- **Toggle actions:** napari automatically creates Window menu entries for all contributed widgets (declared in `napari.yaml` under `contributions.widgets`).
- **"New Lineage Panel...":** Declared as a `napari/window` menu command in `napari.yaml`.

**Action:** Delete `_add_panel_menu_actions()` and the call to it in `launch()`. Delete `_on_new_lineage_panel()` (moved to `_plugin.py`).

**Functionality impact:** Identical behavior. napari's plugin framework provides the same toggle actions that we were manually creating via `toggleViewAction()`.

---

### Step 6: Convert keyboard shortcuts to npe2 keybindings

**Files:** `gui/app.py` (modified), `napari.yaml` (already done in Step 1)

Currently `_bind_keys()` calls `self.viewer.bind_key()` 8 times. In plugin mode, keybindings are declared in `napari.yaml` and napari binds them automatically.

However, `viewer.bind_key()` also works fine in plugins — it's a public API. The npe2 keybinding contributions are optional and mainly useful for discoverability. Since we are not publishing to napari-hub yet, we can **keep `_bind_keys()` as-is** for now. This is a public API with no compatibility risk.

**Action:** No change required. The `napari.yaml` keybindings section can be populated later when publishing.

**Functionality impact:** None.

---

### Step 7: Add config loading dialog for plugin mode

**Files:** `gui/_plugin.py` (modified from Step 2)

In standalone mode, the config path comes from the CLI argument. In plugin mode, the user needs a dialog:

```python
def open_config_dialog(viewer: napari.Viewer):
    from qtpy.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(
        None, "Open AceTree Config", "", "XML files (*.xml);;All files (*)"
    )
    if not path:
        return

    from ..io.config import load_config
    from ..core.nuclei_manager import NucleiManager
    from .app import AceTreeApp

    cfg = load_config(path)
    mgr = NucleiManager.from_config(cfg)
    mgr.process()

    app = AceTreeApp(mgr, viewer=viewer)
    app.launch()
```

This appears in napari's **File** menu as "Open AceTree Config...".

**Functionality impact:** New functionality for plugin mode. No change to existing standalone behavior.

---

### Step 8: Update `pyproject.toml` for dual-mode installation

**Files:** `pyproject.toml` (modified)

```toml
[project.entry-points."napari.manifest"]
acetree-py = "acetree_py:napari.yaml"

[project.scripts]
acetree-py = "acetree_py.__main__:app"
```

Both entry points coexist. `pip install -e ".[gui]"` makes AceTree-Py available as both:
- A CLI tool: `acetree-py gui config.xml`
- A napari plugin: visible in napari's Plugins menu after installation

**Functionality impact:** None to existing users. The CLI works identically. napari users get an additional activation path.

---

## 5. Files Changed Summary

| File | Change type | Description |
|------|------------|-------------|
| `napari.yaml` | **New** | npe2 plugin manifest |
| `gui/_plugin.py` | **New** | Plugin bridge (factory functions, config dialog) |
| `gui/app.py` | **Modified** | Accept `viewer` param; make `napari.Viewer()` creation conditional; replace `_qt_window` dialog parents with `None`; delete `_add_panel_menu_actions()` |
| `pyproject.toml` | **Modified** | Add `napari.manifest` entry point |

| File | Change type | Description |
|------|------------|-------------|
| `gui/viewer_integration.py` | **Unchanged** | All public APIs, works in both modes |
| `gui/contrast_tools.py` | **Unchanged** | All public APIs, works in both modes |
| `gui/lineage_widget.py` | **Unchanged** | Pure Qt, no napari imports |
| `gui/lineage_list.py` | **Unchanged** | Pure Qt, no napari imports |
| `gui/lineage_layout.py` | **Unchanged** | Pure computation, no Qt or napari |
| `gui/edit_panel.py` | **Unchanged** | Pure Qt, no napari imports |
| `gui/cell_info_panel.py` | **Unchanged** | Pure Qt, no napari imports |
| `gui/player_controls.py` | **Unchanged** | Pure Qt, no napari imports |
| `__main__.py` | **Unchanged** | CLI entry point preserved |
| `core/*`, `naming/*`, `editing/*`, `io/*`, `analysis/*` | **Unchanged** | Zero GUI dependencies |

---

## 6. Behavioral Differences: Standalone vs Plugin

| Aspect | Standalone (`main`) | Plugin (`plugin-variant`) |
|--------|-------------------|--------------------------|
| **Launch** | `acetree-py gui config.xml` | napari Plugins menu or File > Open AceTree Config |
| **Viewer ownership** | AceTreeApp creates `napari.Viewer` | AceTreeApp receives existing `Viewer` |
| **Event loop** | `napari.run()` called by `AceTreeApp.run()` | Already running (napari started it) |
| **Save/error dialogs** | Parented to napari window (via `_qt_window`) | Top-level (parent=None) |
| **Window menu toggles** | Manual `toggleViewAction()` via `_dock_widgets` | Automatic via npe2 widget contributions |
| **Keyboard shortcuts** | `viewer.bind_key()` in `_bind_keys()` | Same (`bind_key` is public API) |
| **CLI commands** | All work (load, export, rename, info, gui) | All work identically |
| **Multiple configs** | One per process | Could open multiple configs in one napari session (future work) |

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| npe2 manifest schema changes in future napari | Low | Medium | Pin napari `>=0.5,<0.7` as already done |
| `viewer.bind_key()` conflicts with other plugins | Low | Low | Keys are standard (arrows, Ctrl+S/Z/Y); unlikely to conflict |
| Parentless dialogs feel disconnected on multi-monitor setups | Low | Low | Qt centers on screen; acceptable UX |
| `add_dock_widget()` behavior changes | Low | Medium | Public API, well-tested, unlikely to break |
| Singleton `_app_instance` in `_plugin.py` causes issues with multiple viewer windows | Medium | Medium | Check `viewer` identity before reusing; create new instance if viewer changed |

---

## 8. Testing Plan

All existing tests (557) must continue to pass on the `plugin-variant` branch without modification, since the standalone code path is preserved.

Additional plugin-specific tests:

| Test | Description |
|------|-------------|
| `test_app_accepts_viewer` | `AceTreeApp(mgr, viewer=existing_viewer)` doesn't create a new viewer |
| `test_app_standalone_creates_viewer` | `AceTreeApp(mgr)` + `launch()` creates viewer (existing behavior) |
| `test_plugin_factory_creates_app` | `_plugin.create_main_widget(viewer)` returns a valid QWidget |
| `test_config_dialog_loads_data` | `open_config_dialog` with a test config file creates a working app |
| `test_dialogs_work_without_parent` | `QFileDialog(None, ...)` and `QMessageBox(None, ...)` function correctly |

---

## 9. Migration Order

Execute steps in this order to maintain a working codebase at each stage:

1. **Create branch** `plugin-variant` off `main`
2. **Step 3** — Refactor `AceTreeApp.__init__` (backward-compatible, no breakage)
3. **Step 4** — Replace `_qt_window` dialog parents (no breakage)
4. **Step 5** — Delete `_add_panel_menu_actions()` (standalone Window menu entries removed; this is the only intentional behavior change in standalone mode — dock widgets will no longer appear in the Window menu, but they were only recently added and are not critical in standalone mode)
5. **Step 1** — Add `napari.yaml` manifest
6. **Step 2** — Create `_plugin.py` bridge module
7. **Step 7** — Add config loading dialog
8. **Step 8** — Add entry point to `pyproject.toml`
9. **Run full test suite** — verify 557+ tests pass
10. **Manual testing** — verify both `acetree-py gui config.xml` and napari plugin activation work

---

## 10. Estimated Effort

| Step | LOC changed | Time |
|------|------------|------|
| Step 1: napari.yaml | ~40 new | 15 min |
| Step 2: _plugin.py | ~80 new | 30 min |
| Step 3: Accept viewer | ~15 changed | 15 min |
| Step 4: Remove _qt_window | ~6 changed | 10 min |
| Step 5: Remove _dock_widgets | ~30 deleted | 10 min |
| Step 6: Keybindings | 0 (no change) | 0 |
| Step 7: Config dialog | ~30 new | 20 min |
| Step 8: pyproject.toml | ~2 added | 5 min |
| Testing | ~40 new | 30 min |
| **Total** | ~200 net | ~2-3 hours |
