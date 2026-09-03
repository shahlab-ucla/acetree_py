# Bundled StarryNite assets

These files come from [zhirongbaolab/StarryNite](https://github.com/zhirongbaolab/StarryNite), revision `e3d5ddc381223ae8ce031f64947ffbd944a11593`.

They are redistributed under the upstream GNU GPL v3 license in
`LICENSE.GPL-3.0.txt`. The surrounding AceTree-Python source remains under its
repository license; these third-party files retain their upstream license and
provenance.

Included upstream data:

- every text parameter file from `example_parameter_files/newmatlab`;
- `distribution_lineaging/2019TrackingModelv2.mat`;
- `example_parameter_files/newmatlab/gaussianlatedispimmodel_withoptimizedfeatures_ignoringFPstillpoorFN.mat`;
- `distribution_lineaging/clean_red_singlemodel_red_normal.mat` (pre-2019
  source model; old-MATLAB export required);
- `distribution_lineaging/clean_distributions_newimage.mat`; and
- `distribution_code/clean_distributions_newimage_10thround_edited2.mat`.

The install-ready parameter copies are marked modifications: only model and
detector-distribution paths were changed from machine-specific or sibling-file
paths to package-relative `../models` and `../distributions` paths. Biological
and tracking parameter values were not changed. The
`clean_distributions_newimage_10thround.mat` file is a byte-identical filename
alias of the upstream `clean_distributions_newimage_10thround_edited2.mat`,
because the upstream iSIM/SD parameter files reference the shorter name.

The `.atpy-model` files are deterministic numeric exports produced from the
adjacent MAT files by AceTree-Python's source-bound MATLAB exporter using
MATLAB R2025a. They contain no executable MATLAB objects and embed the SHA-256
of their source MAT file. Users of these bundled models do not need to create
or select JSON files.

The historical red-channel `NaiveBayes` source model is included, but is not
bundled as a numeric export because current MATLAB reconstructs that retired
object as empty. Use
`acetree-starrynite-export-model` with an older compatible MATLAB release; the
underlying helper is
`acetree_py/tracking/starrynite/oracle/matlab/export_starrynite_classifier_numeric.m`.
