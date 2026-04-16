"""CLI entry point for AceTree-Py.

Provides commands for loading datasets, launching the GUI, exporting
data, running the naming pipeline, and querying cell information.

Usage:
    acetree load <config.xml>
    acetree gui <config.xml>
    acetree export <config.xml> --format csv --output cells.csv
    acetree rename <config.xml> --output renamed.zip
    acetree info <config.xml> --cell ABala
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Redirect numba cache to a short path BEFORE any napari/numba imports.
# The Windows Store Python install path can be very long (~200 chars), and
# numba's temp filenames push the total over the 260-char Windows MAX_PATH limit.
if not os.environ.get("NUMBA_CACHE_DIR"):
    _cache = os.path.join(os.path.expanduser("~"), ".acetree_cache", "numba")
    os.makedirs(_cache, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = _cache
    del _cache

import typer

app = typer.Typer(
    help="AceTree-Py: C. elegans embryogenesis visualization and annotation.",
    no_args_is_help=True,
)


def _load_manager(config_path: str):
    """Load a NucleiManager from a config file."""
    from acetree_py.io.config import load_config
    from acetree_py.core.nuclei_manager import NucleiManager

    cfg = load_config(Path(config_path))
    mgr = NucleiManager.from_config(cfg)
    mgr.process()
    return mgr


@app.command()
def load(config: str = typer.Argument(..., help="Path to XML config file")):
    """Load a dataset and print summary information."""
    mgr = _load_manager(config)

    num_nuclei = sum(len(tp) for tp in mgr.nuclei_record)
    num_alive = sum(
        sum(1 for n in tp if n.is_alive) for tp in mgr.nuclei_record
    )
    num_cells = mgr.lineage_tree.num_cells if mgr.lineage_tree else 0

    typer.echo(f"Dataset: {config}")
    typer.echo(f"  Timepoints:    {mgr.num_timepoints}")
    typer.echo(f"  Total nuclei:  {num_nuclei}")
    typer.echo(f"  Alive nuclei:  {num_alive}")
    typer.echo(f"  Lineage cells: {num_cells}")

    if mgr.lineage_tree and mgr.lineage_tree.root:
        typer.echo(f"  Root cell:     {mgr.lineage_tree.root.name}")


@app.command()
def gui(config: str = typer.Argument(..., help="Path to XML config file")):
    """Launch the napari GUI for interactive visualization and editing."""
    from acetree_py.gui.app import AceTreeApp

    ace = AceTreeApp.from_config(config)
    ace.run()


@app.command()
def export(
    config: str = typer.Argument(..., help="Path to XML config file"),
    format: str = typer.Option(
        "cell_csv", "--format", "-f",
        help="Export format: cell_csv, nucleus_csv, expression_csv, newick",
    ),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output file path (default: auto-named based on format)",
    ),
):
    """Export lineage data to CSV or Newick format."""
    mgr = _load_manager(config)

    if mgr.lineage_tree is None:
        typer.echo("Error: No lineage tree built.", err=True)
        raise typer.Exit(1)

    # Auto-name output if not specified
    config_stem = Path(config).stem
    if output is None:
        extensions = {
            "cell_csv": f"{config_stem}_cells.csv",
            "nucleus_csv": f"{config_stem}_nuclei.csv",
            "expression_csv": f"{config_stem}_expression.csv",
            "newick": f"{config_stem}_tree.nwk",
        }
        output = extensions.get(format, f"{config_stem}_export.txt")

    from acetree_py.analysis.export import (
        export_cell_table_csv,
        export_expression_csv,
        export_newick,
        export_nucleus_table_csv,
    )

    if format == "cell_csv":
        export_cell_table_csv(mgr.lineage_tree, output)
    elif format == "nucleus_csv":
        export_nucleus_table_csv(mgr.nuclei_record, output)
    elif format == "expression_csv":
        export_expression_csv(mgr.lineage_tree, output)
    elif format == "newick":
        export_newick(mgr.lineage_tree, output)
    else:
        typer.echo(f"Unknown format: {format}. Use: cell_csv, nucleus_csv, expression_csv, newick", err=True)
        raise typer.Exit(1)

    typer.echo(f"Exported {format} to {output}")


@app.command()
def rename(
    config: str = typer.Argument(..., help="Path to XML config file"),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output ZIP path (default: <input>_renamed.zip)",
    ),
):
    """Run the naming pipeline and save renamed nuclei."""
    mgr = _load_manager(config)

    if output is None:
        output = str(Path(config).with_suffix("").with_name(
            Path(config).stem + "_renamed.zip"
        ))

    mgr.save(Path(output))
    typer.echo(f"Saved renamed nuclei to {output}")


@app.command()
def info(
    config: str = typer.Argument(..., help="Path to XML config file"),
    cell: str = typer.Option(..., "--cell", "-c", help="Cell name to query"),
):
    """Print detailed information about a specific cell."""
    mgr = _load_manager(config)
    cell_obj = mgr.get_cell(cell)

    if cell_obj is None:
        # Try case-insensitive
        if mgr.lineage_tree:
            cell_obj = mgr.lineage_tree.get_cell_icase(cell)

    if cell_obj is None:
        typer.echo(f"Cell '{cell}' not found in lineage tree.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Cell: {cell_obj.name}")
    typer.echo(f"  Lifetime:  t={cell_obj.start_time} - {cell_obj.end_time} ({cell_obj.lifetime} timepoints)")
    typer.echo(f"  Fate:      {cell_obj.end_fate.name}")
    typer.echo(f"  Depth:     {cell_obj.depth()} divisions from root")

    if cell_obj.parent:
        typer.echo(f"  Parent:    {cell_obj.parent.name}")
    if cell_obj.children:
        names = ", ".join(c.name for c in cell_obj.children)
        typer.echo(f"  Children:  {names}")

    # Expression summary
    from acetree_py.analysis.expression import cell_expression_time_series

    ts = cell_expression_time_series(cell_obj)
    if ts.values:
        typer.echo(f"  Expression:")
        typer.echo(f"    Mean:    {ts.mean:.2f}")
        typer.echo(f"    Max:     {ts.max_value:.2f}")
        typer.echo(f"    Onset:   {ts.onset_time or 'never'}")

    # Position at first timepoint
    if cell_obj.nuclei:
        _, nuc = cell_obj.nuclei[0]
        typer.echo(f"  Position (t={cell_obj.start_time}): ({nuc.x}, {nuc.y}, {nuc.z:.1f})")
        typer.echo(f"  Size:      {nuc.size}")

    # Subtree info
    num_descendants = sum(1 for _ in cell_obj.iter_descendants())
    num_leaves = sum(1 for _ in cell_obj.iter_leaves())
    typer.echo(f"  Descendants: {num_descendants}")
    typer.echo(f"  Leaves:      {num_leaves}")


@app.command()
def create(
    directory: Optional[str] = typer.Argument(
        None, help="Path to image directory (omit for interactive dialog)",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output directory for nuclei ZIP + config XML",
    ),
    xy_res: float = typer.Option(0.09, "--xy-res", help="XY pixel resolution (µm)"),
    z_res: float = typer.Option(1.0, "--z-res", help="Z pixel resolution (µm)"),
    split: bool = typer.Option(False, "--split", help="Split side-by-side channels"),
    flip: bool = typer.Option(False, "--flip", help="Flip image left/right"),
    interleaved: bool = typer.Option(
        False, "--interleaved",
        help="Single TIFF per timepoint contains interleaved multichannel pages",
    ),
    num_channels: int = typer.Option(
        1, "--num-channels",
        help="Number of channels (required with --interleaved, must be >= 2)",
    ),
    channel_order: str = typer.Option(
        "CZ", "--channel-order",
        help="Page order for interleaved stacks: 'CZ' (channel-fastest) or 'ZC' (planar)",
    ),
):
    """Create a new dataset from raw images and launch the GUI for manual annotation."""
    from acetree_py.gui.app import AceTreeApp

    if directory is None:
        # Interactive mode: show dataset creation dialog
        ace = AceTreeApp.from_dialog()
        if ace is None:
            typer.echo("Dataset creation cancelled.")
            raise typer.Exit(0)
        ace.run()
    else:
        # CLI mode: build config from arguments
        from acetree_py.io.config import AceTreeConfig, NamingMethod, _derive_image_params

        img_dir = Path(directory).resolve()
        if not img_dir.is_dir():
            typer.echo(f"Error: '{directory}' is not a directory.", err=True)
            raise typer.Exit(1)

        out_dir = Path(output).resolve() if output else img_dir / "acetree_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect image count and planes
        tiffs = sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.tiff"))
        if not tiffs:
            typer.echo(f"Error: No TIFF files found in '{directory}'.", err=True)
            raise typer.Exit(1)

        # Probe first image for page count
        num_pages = 1
        try:
            import tifffile
            with tifffile.TiffFile(tiffs[0]) as tf:
                num_pages = len(tf.pages)
        except Exception:
            pass

        # Validate interleaved args and compute real Z-plane count
        effective_channels = 1
        effective_order = "CZ"
        if interleaved:
            if num_channels < 2:
                typer.echo(
                    "Error: --interleaved requires --num-channels >= 2.",
                    err=True,
                )
                raise typer.Exit(1)
            order = (channel_order or "CZ").strip().upper()
            if order not in ("CZ", "ZC"):
                typer.echo(
                    f"Error: --channel-order must be 'CZ' or 'ZC' (got '{channel_order}').",
                    err=True,
                )
                raise typer.Exit(1)
            effective_channels = num_channels
            effective_order = order
            # Interleaved bypasses the split/flip wrapper — warn if user set both
            if split or flip:
                typer.echo(
                    "Warning: --split/--flip are ignored when --interleaved is set.",
                    err=True,
                )
            split = False
            flip = False
            num_planes = max(1, num_pages // num_channels)
        else:
            num_planes = num_pages

        num_timepoints = len(tiffs)
        dataset_name = img_dir.name

        # Use first TIFF as the typical image_file so _derive_image_params
        # can extract tif_directory and tif_prefix automatically.
        config = AceTreeConfig(
            config_file=out_dir / f"{dataset_name}.xml",
            image_file=tiffs[0],
            zip_file=out_dir / f"{dataset_name}_nuclei.zip",
            num_channels=effective_channels,
            xy_res=xy_res,
            z_res=z_res,
            plane_end=num_planes,
            starting_index=1,
            ending_index=num_timepoints,
            naming_method=NamingMethod.NEWCANONICAL,
            expr_corr="none",
            use_zip=0,
            use_stack=1 if num_planes > 1 else 0,
            split=1 if split else 0,
            flip=1 if flip else 0,
            stack_interleaved=interleaved,
            stack_channel_order=effective_order,
        )
        _derive_image_params(config)

        ace = AceTreeApp.from_new_dataset(config, num_timepoints, out_dir)
        ace.run()


if __name__ == "__main__":
    app()
