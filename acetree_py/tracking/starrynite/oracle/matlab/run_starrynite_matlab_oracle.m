function run_starrynite_matlab_oracle(request_path, result_path, starrynite_root)
%RUN_STARRYNITE_MATLAB_ORACLE Call low-level functions from a StarryNite checkout.
%
% This is a test-only interoperability adapter. It does not contain a copy of
% StarryNite's filtering or candidate-detection implementation; the functions
% are resolved from the user-supplied STARRYNITE_ROOT/distribution_code folder.
%
% Inputs
% ------
% request_path:
%   A MATLAB v7 MAT-file containing one scalar struct named "request".
% result_path:
%   Destination MATLAB v7 MAT-file. It contains one scalar struct named
%   "result", including structured error information if an operation fails.
% starrynite_root:
%   Absolute path to a checkout of zhirongbaolab/StarryNite.
%
% Orientation contract
% --------------------
% Input volumes use MATLAB image order [Y, X, Z]. Python callers with [Z, Y,
% X] arrays must transpose with (1, 2, 0) before writing the request MAT-file.
% StarryNite candidate coordinates use one-based [X, Y, Z]. This adapter also
% returns zero-based [Z, Y, X] fields for unambiguous Python comparisons.
%
% Operations (request.operation)
% ------------------------------
% "gaussian_filter":
%   volume_yxz, sigma_yxz, kernel_size_yxz
% "separable_dog":
%   volume_yxz, inner_sigma_yxz, inner_kernel_size_yxz,
%   outer_sigma_yxz, outer_kernel_size_yxz
% "tiled_dog":
%   volume_yxz, sigma_xy, anisotropy
% "resolve_parameter":
%   parameter_name, staging, parameter_values, num_cells, location_xyz;
%   optional regional_stage_index, regional_area, regional_value. Invokes
%   upstream getParameter without evaluating a parameter file.
% "slice_candidates":
%   filtered_volume_yxz, maxima_threshold, cell_diameter_xy, anisotropy,
%   num_cells, legacy_parameters; optional z_level, roi_points_xy,
%   roi_is_local, roi_x_min, roi_y_min. legacy_parameters is assigned to the
%   global expected by StarryNite's getParameter function.
% "full_detection":
%   volume_yxz, cell_diameter_xy, anisotropy, num_cells,
%   legacy_parameters; optional distribution_file. This initializes the
%   script workspace required by processVolume.m without evaluating a legacy
%   parameter file and returns its final and diagnostic detection artifacts.
% "full_tracking":
%   movie_yxzt, cell_diameter_xy, anisotropy, num_cells,
%   legacy_parameters; optional distribution_file, model_file, and scalar
%   tracking_overrides; use_static_diameter is required. Each frame is
%   detected sequentially by processVolume before the original 2019
%   classifier-based lineage driver is invoked.
% "export_classifier_model":
%   Optional model_file. Loads a ClassificationNaiveBayes model from the
%   external checkout and returns only neutral numeric, logical, text, cell,
%   and scalar-struct data. Kernel distributions are expanded to bandwidth,
%   support, and InputData arrays; no executable MATLAB object is returned.
% "predict_bifurcation":
%   daughter_data (1-by-22), back_data (1-by-11), forward_data (1-by-13),
%   branch/topology scalars, force_mode, and optional model_file. Invokes the
%   upstream predictBifurcationTypeSinglemodel function.
%
% "separable_dog" deliberately accepts the two Gaussian parameter triples.
% The caller derives them from the legacy parameter set. This keeps the
% adapter independent of StarryNite parameter policy while still executing
% StarryNite's own numerical filter implementation.

    % StarryNite's getParameter.m API requires this global workspace variable.
    global parameters; %#ok<GVMIS>
    global computedclassificationvector; %#ok<GVMIS>
    global refclassificationvector; %#ok<GVMIS>
    global removed; %#ok<GVMIS>
    global simpleFNcorrect; %#ok<GVMIS>
    global FNtype; %#ok<GVMIS>
    global classround; %#ok<GVMIS>
    global ATPY_STARRYNITE_EVENT_TRACE; %#ok<GVMIS>
    global ATPY_STARRYNITE_STAGE_TRACE; %#ok<GVMIS>

    request_path = require_path_text(request_path, 'request_path');
    result_path = require_path_text(result_path, 'result_path');
    starrynite_root = require_path_text(starrynite_root, 'starrynite_root');

    result = struct();
    result.schema_version = uint32(1);
    result.success = false;
    result.request_path = request_path;
    result.starrynite_root = starrynite_root;
    result.matlab_version = version;
    started = tic;

    try
        distribution_dir = fullfile(starrynite_root, 'distribution_code');
        if ~isfolder(distribution_dir)
            error('ATPy:StarryNiteOracle:MissingDistributionCode', ...
                'StarryNite distribution_code folder not found: %s', distribution_dir);
        end
        distribution_dir = absolute_existing_path(distribution_dir);
        addpath(distribution_dir, '-begin');
        path_cleanup = onCleanup(@() rmpath(distribution_dir));

        loaded = load(request_path, 'request');
        if ~isfield(loaded, 'request') || ~isstruct(loaded.request) || ...
                ~isscalar(loaded.request)
            error('ATPy:StarryNiteOracle:InvalidRequest', ...
                'Request MAT-file must contain one scalar struct named request.');
        end
        request = loaded.request;
        require_schema_version(request);
        operation = require_text_field(request, 'operation');

        result.operation = operation;
        result.distribution_code_path = distribution_dir;
        result.image_processing_toolbox_available = ~isempty(ver('images'));

        switch operation
            case 'resolve_parameter'
                require_upstream_function('getParameter', distribution_dir);
                parameter_name = require_text_field(request, 'parameter_name');
                if ~isvarname(parameter_name)
                    error('ATPy:StarryNiteOracle:InvalidField', ...
                        'request.parameter_name must be a direct MATLAB struct field.');
                end
                staging = require_finite_vector(request, 'staging', 0);
                parameter_values = require_finite_vector( ...
                    request, 'parameter_values', 0);
                num_cells = require_finite_scalar(request, 'num_cells');
                location = require_finite_vector(request, 'location_xyz', 3);

                previous_parameters = parameters;
                parameters = struct();
                parameters.staging = staging;
                parameters.(parameter_name) = parameter_values;
                parameter_cleanup = onCleanup( ...
                    @() restore_parameters(previous_parameters));

                regional_fields = { ...
                    'regional_stage_index', 'regional_area', 'regional_value'};
                regional_present = cellfun( ...
                    @(name) isfield(request, name), regional_fields);
                if any(regional_present) && ~all(regional_present)
                    error('ATPy:StarryNiteOracle:InvalidField', ...
                        ['regional_stage_index, regional_area, and ', ...
                         'regional_value must be supplied together.']);
                end
                if all(regional_present)
                    regional_stage = require_positive_scalar( ...
                        request, 'regional_stage_index');
                    if regional_stage ~= floor(regional_stage)
                        error('ATPy:StarryNiteOracle:InvalidField', ...
                            'request.regional_stage_index must be an integer.');
                    end
                    regional_area = require_finite_vector( ...
                        request, 'regional_area', 6);
                    regional_value = require_finite_vector( ...
                        request, 'regional_value', 0);
                    parameters.regions = cell(regional_stage, 1);
                    parameters.regions{regional_stage}.area = regional_area;
                    parameters.regions{regional_stage}.(parameter_name) = ...
                        regional_value;
                end

                resolved = getParameter(parameter_name, num_cells, location);
                if ~isnumeric(resolved) || ~isreal(resolved) || ...
                        isempty(resolved) || any(~isfinite(resolved(:)))
                    error('ATPy:StarryNiteOracle:InvalidResult', ...
                        'getParameter returned a non-finite or nonnumeric value.');
                end
                result.upstream_function = which('getParameter');
                result.parameter_name = parameter_name;
                result.num_cells = num_cells;
                result.location_xyz = location;
                result.resolved_parameter = double(resolved);

            case 'gaussian_filter'
                require_upstream_function('imgaussianAnisotropy', distribution_dir);
                volume = require_volume(request, 'volume_yxz');
                sigma = require_axis_vector(request, 'sigma_yxz', true);
                kernel_size = require_axis_vector(request, 'kernel_size_yxz', false);

                result.input_shape_yxz = uint32(size(volume));
                result.upstream_function = which('imgaussianAnisotropy');
                result.filtered_volume_yxz = imgaussianAnisotropy( ...
                    single(volume), sigma, kernel_size);

            case 'separable_dog'
                require_upstream_function('imgaussianAnisotropy', distribution_dir);
                volume = require_volume(request, 'volume_yxz');
                inner_sigma = require_axis_vector(request, 'inner_sigma_yxz', true);
                inner_size = require_axis_vector( ...
                    request, 'inner_kernel_size_yxz', false);
                outer_sigma = require_axis_vector(request, 'outer_sigma_yxz', true);
                outer_size = require_axis_vector( ...
                    request, 'outer_kernel_size_yxz', false);

                result.input_shape_yxz = uint32(size(volume));
                result.upstream_function = which('imgaussianAnisotropy');
                inner = imgaussianAnisotropy(single(volume), inner_sigma, inner_size);
                outer = imgaussianAnisotropy(single(volume), outer_sigma, outer_size);
                result.filtered_volume_yxz = inner - outer;

            case 'tiled_dog'
                require_upstream_function('tiledogfilter', distribution_dir);
                volume = require_volume(request, 'volume_yxz');
                sigma_xy = require_positive_scalar(request, 'sigma_xy');
                anisotropy = require_positive_scalar(request, 'anisotropy');

                result.input_shape_yxz = uint32(size(volume));
                result.upstream_function = which('tiledogfilter');
                result.filtered_volume_yxz = tiledogfilter( ...
                    single(volume), sigma_xy, anisotropy);

            case 'slice_candidates'
                require_upstream_function('createDiskSet', distribution_dir);
                require_upstream_function('getParameter', distribution_dir);
                volume = require_volume(request, 'filtered_volume_yxz');
                maxima_threshold = require_finite_scalar(request, 'maxima_threshold');
                cell_diameter = require_positive_scalar(request, 'cell_diameter_xy');
                anisotropy = require_positive_scalar(request, 'anisotropy');
                num_cells = require_nonnegative_scalar(request, 'num_cells');
                legacy_parameters = require_legacy_parameters(request);
                z_level = optional_scalar(request, 'z_level', size(volume, 3));
                roi_points = optional_roi_points(request);
                roi_is_local = logical(optional_scalar(request, 'roi_is_local', 0));
                roi_x_min = optional_scalar(request, 'roi_x_min', 0);
                roi_y_min = optional_scalar(request, 'roi_y_min', 0);

                previous_parameters = parameters;
                parameters = legacy_parameters;
                parameter_cleanup = onCleanup( ...
                    @() restore_parameters(previous_parameters));

                [disk_set, center_indices] = createDiskSet( ...
                    single(volume), maxima_threshold, z_level, cell_diameter, ...
                    anisotropy, num_cells, roi_points, roi_is_local, ...
                    roi_x_min, roi_y_min);

                slice_maxima = as_xyz_table(disk_set.xymax, 'diskSet.xymax');
                centered_maxima = as_xyz_table( ...
                    disk_set.centeredxymax, 'diskSet.centeredxymax');
                center_indices = double(center_indices(:));
                validate_indices(center_indices, size(centered_maxima, 1));

                result.input_shape_yxz = uint32(size(volume));
                result.upstream_function = which('createDiskSet');
                result.upstream_parameter_function = which('getParameter');
                result.legacy_parameters = legacy_parameters;
                result.slice_maxima_xyz_1based = slice_maxima;
                result.slice_maxima_zyx_0based = xyz_one_to_zyx_zero(slice_maxima);
                result.centered_slice_maxima_xyz_1based = centered_maxima;
                result.centered_slice_maxima_zyx_0based = ...
                    xyz_one_to_zyx_zero(centered_maxima);
                result.center_indices_1based = center_indices;
                result.candidate_centers_xyz_1based = centered_maxima(center_indices, :);
                result.candidate_centers_zyx_0based = xyz_one_to_zyx_zero( ...
                    centered_maxima(center_indices, :));
                result.candidate_maxima = select_vector_field( ...
                    disk_set, 'xymaximavals', center_indices);
                result.candidate_diameters_xy = select_vector_field( ...
                    disk_set, 'xydetdiameters', center_indices);
                result.candidate_xy_coverage = select_vector_field( ...
                    disk_set, 'xycoverage', center_indices);

            case 'full_detection'
                require_upstream_function('processVolume', distribution_dir);
                require_upstream_function('getParameter', distribution_dir);
                volume = require_volume(request, 'volume_yxz');
                cell_diameter = require_positive_scalar(request, 'cell_diameter_xy');
                anisotropy = require_positive_scalar( ...
                    request, 'anisotropy'); %#ok<NASGU>
                num_cells = require_nonnegative_scalar(request, 'num_cells');
                legacy_parameters = require_detector_legacy_parameters(request);
                distribution_file = optional_distribution_file( ...
                    request, distribution_dir);

                previous_parameters = parameters;
                parameters = legacy_parameters;
                parameter_cleanup = onCleanup( ...
                    @() restore_parameters(previous_parameters));

                % processVolume is a legacy script rather than a function. The
                % variables below are the minimal no-label, static-diameter
                % workspace used by its production all-MATLAB driver.
                X = single(volume); %#ok<NASGU>
                nodatause = true; %#ok<NASGU>
                nodata = true; %#ok<NASGU>
                previous = 0; %#ok<NASGU>
                usestaticdiameter = true; %#ok<NASGU>
                firsttimestepdiam = cell_diameter; %#ok<NASGU>
                downsample = 1; %#ok<NASGU>
                firsttimestepnumcells = num_cells; %#ok<NASGU>
                singlevolume = false; %#ok<NASGU>
                % The legacy scan only assigns zlevel when some plane exceeds
                % 1.5 times the maxima threshold. Keep a deterministic fallback
                % for empty/high-threshold synthetic trials; createDiskSet's
                % current production path does not otherwise consume zlevel.
                zlevel = size(volume, 3); %#ok<NASGU>
                ROIpoints = zeros(0, 2); %#ok<PREALL>
                ROI = false; %#ok<NASGU>
                ROIxmin = 1; %#ok<NASGU>
                ROIymin = 1; %#ok<NASGU>
                savedata = true; %#ok<NASGU>
                conservememory = false; %#ok<NASGU>
                e = struct();
                load(distribution_file); %#ok<LOAD>
                processVolume;

                final_points = as_xyz_table(e.finalpoints, 'e.finalpoints');
                average_points = as_xyz_table( ...
                    e.finalaveragepoints, 'e.finalaveragepoints');
                result.input_shape_yxz = uint32(size(volume));
                result.upstream_function = which('processVolume');
                result.distribution_file = distribution_file;
                result.legacy_parameters = legacy_parameters;
                result.final_points_xyz_1based = final_points;
                result.final_points_zyx_0based = ...
                    xyz_one_to_zyx_zero(final_points);
                result.final_average_points_xyz_1based = average_points;
                result.final_average_points_zyx_0based = ...
                    xyz_one_to_zyx_zero(average_points);
                result.final_diameters_xy = double(e.finaldiams(:));
                result.final_maxima = double(e.finalmaximas(:));
                result.effective_sigma_xy = double(e.sigma);
                result.effective_cell_diameter_xy = double(e.celldiameter);
                result.effective_num_cells = double(e.numcells);
                result = copy_optional_detection_fields(result, e);
                result.raw_disk_xycoverage = double(diskSet.xycoverage(:));
                result.raw_disk_diameters_xy = ...
                    double(diskSet.xydetdiameters(:));
                result.nucleus_center_indices_1based = ...
                    double(nucleiSet.centerindicies(:));
                result.nucleus_log_odds = nucleiSet.logodds;
                result.nucleus_ranges_1based = nucleiSet.range;
                result.nucleus_assigned_planes = nucleiSet.centers;
                result.nucleus_disk_features = cell(size(nucleiSet.centers));
                result.nucleus_serial_log_odds = cell(size(nucleiSet.centers));
                for nucleus_index = 1:numel(nucleiSet.centers)
                    center_disk = nucleiSet.centerindicies(nucleus_index);
                    result.nucleus_disk_features{nucleus_index} = ...
                        calc_disk_feature_vector( ...
                            nucleiSet.centers{nucleus_index}, ...
                            diskSet.xymaximavals(center_disk), ...
                            diskSet.xydetdiameters(center_disk), ...
                            diskSet.centeredxymax(center_disk, :), ...
                            diskSet.xymaximavals, ...
                            diskSet.xydetdiameters, anisotropy);
                    result.nucleus_serial_log_odds{nucleus_index} = ...
                        calculateLogodds( ...
                            nucleiSet.centers{nucleus_index}, ...
                            diskSet.centeredxymax(center_disk, :), ...
                            diskSet.xymaximavals(center_disk), ...
                            diskSet.xydetdiameters(center_disk), ...
                            diskSet.xymaximavals, ...
                            diskSet.xydetdiameters, anisotropy, ...
                            diskSet.xycoverage);
                end

            case 'export_classifier_model'
                lineaging_dir = require_lineaging_directory(starrynite_root);
                addpath(lineaging_dir, '-begin');
                lineaging_cleanup = onCleanup(@() rmpath(lineaging_dir));
                [trackingparameters, classifier_model, model_file] = ...
                    load_tracking_classifier(request, lineaging_dir);

                result.upstream_function = ...
                    'ClassificationNaiveBayes property export';
                result.model_file = model_file;
                result.source_model = source_model_metadata( ...
                    model_file, classifier_model);
                result.classifier_model = neutral_classifier_model( ...
                    trackingparameters, classifier_model);

            case 'predict_bifurcation'
                lineaging_dir = require_lineaging_directory(starrynite_root);
                addpath(lineaging_dir, '-begin');
                lineaging_cleanup = onCleanup(@() rmpath(lineaging_dir));
                require_upstream_function( ...
                    'predictBifurcationTypeSinglemodel', lineaging_dir);
                [trackingparameters, classifier_model, model_file] = ...
                    load_tracking_classifier(request, lineaging_dir);
                trackingparameters.trainingmode = false;
                trackingparameters.useclassifieroracle = false;

                daughter_data = require_feature_row( ...
                    request, 'daughter_data', 22);
                back_data = require_feature_row(request, 'back_data', 11);
                forward_data = require_feature_row( ...
                    request, 'forward_data', 13);
                d1_length = require_positive_scalar(request, 'd1_length');
                d2_length = require_positive_scalar(request, 'd2_length');
                fn_back_1_length = require_finite_scalar( ...
                    request, 'fn_back_candidate_1_length');
                fn_back_2_length = require_finite_scalar( ...
                    request, 'fn_back_candidate_2_length');
                best_forward_d1 = require_finite_scalar( ...
                    request, 'best_fn_forward_length_d1');
                best_forward_d2 = require_finite_scalar( ...
                    request, 'best_fn_forward_length_d2');
                best_back_correct = require_logical_scalar( ...
                    request, 'best_fn_back_correct');
                best_index = require_classifier_index(request, 'best_index');
                force_mode = require_logical_scalar(request, 'force_mode');

                [classifier_input, topology_class, effective_back, ...
                    effective_forward, topology_flags] = ...
                    assemble_singlemodel_classifier_input( ...
                    daughter_data, back_data, forward_data, ...
                    d1_length, d2_length, fn_back_1_length, ...
                    fn_back_2_length, best_forward_d1, best_forward_d2, ...
                    trackingparameters);
                [direct_class, posterior_scores, posterior_classes] = ...
                    direct_classifier_prediction( ...
                    classifier_model, classifier_input);

                previous_computed = computedclassificationvector;
                previous_reference = refclassificationvector;
                previous_removed = removed;
                previous_simple_fn = simpleFNcorrect;
                classifier_cleanup = onCleanup(@() restore_classifier_globals( ...
                    previous_computed, previous_reference, ...
                    previous_removed, previous_simple_fn));
                computedclassificationvector = [];
                refclassificationvector = [];
                removed = [];
                simpleFNcorrect = [];

                predicted_class = predictBifurcationTypeSinglemodel( ...
                    daughter_data, forward_data, back_data, ...
                    d1_length, d2_length, fn_back_1_length, ...
                    fn_back_2_length, best_forward_d1, best_forward_d2, ...
                    best_back_correct, trackingparameters, best_index, ...
                    1, force_mode);
                if numel(computedclassificationvector) ~= 1 || ...
                        numel(refclassificationvector) ~= 1
                    error('ATPy:StarryNiteOracle:UnexpectedClassifierOutput', ...
                        ['predictBifurcationTypeSinglemodel did not record one ', ...
                         'computed and one final classification.']);
                end

                result.upstream_function = which( ...
                    'predictBifurcationTypeSinglemodel');
                result.model_file = model_file;
                result.source_model = source_model_metadata( ...
                    model_file, classifier_model);
                result.predicted_class = double(predicted_class);
                result.computed_class = double(computedclassificationvector);
                result.reference_class = double(refclassificationvector);
                result.direct_predicted_class = double(direct_class);
                result.posterior_scores = double(posterior_scores);
                result.posterior_class_names = double(posterior_classes(:)');
                result.classifier_input = double(classifier_input);
                result.topology_class = double(topology_class);
                result.topology_flags = topology_flags;
                result.effective_back_data = double(effective_back);
                result.effective_forward_data = double(effective_forward);
                result.force_mode = logical(force_mode);
                result.best_index = double(best_index);

            case 'full_tracking'
                require_upstream_function('processVolume', distribution_dir);
                require_upstream_function('getParameter', distribution_dir);
                movie = require_movie(request, 'movie_yxzt');
                cell_diameter = require_positive_scalar(request, 'cell_diameter_xy');
                anisotropy = require_positive_scalar(request, 'anisotropy');
                num_cells = require_nonnegative_scalar(request, 'num_cells');
                use_static_diameter = require_logical_scalar( ...
                    request, 'use_static_diameter');
                legacy_parameters = require_detector_legacy_parameters(request);
                distribution_file = optional_distribution_file( ...
                    request, distribution_dir);
                lineaging_dir = fullfile(starrynite_root, 'distribution_lineaging');
                if ~isfolder(lineaging_dir)
                    error('ATPy:StarryNiteOracle:MissingLineagingCode', ...
                        'StarryNite distribution_lineaging folder not found: %s', ...
                        lineaging_dir);
                end
                lineaging_dir = absolute_existing_path(lineaging_dir);
                addpath(lineaging_dir, '-begin');
                lineaging_cleanup = onCleanup(@() rmpath(lineaging_dir));
                require_upstream_function( ...
                    'tracking_driver_new_classifier_based_version', lineaging_dir);
                upstream_tracking_driver = which( ...
                    'tracking_driver_new_classifier_based_version');
                event_trace_dir = install_tracking_event_trace(lineaging_dir);
                addpath(event_trace_dir, '-begin');
                event_trace_path_cleanup = onCleanup(@() ...
                    cleanup_tracking_event_trace(event_trace_dir));
                rehash path;
                clear tracking_driver_new_classifier_based_version;
                clear greedydeleteFPbranches processOtherBifurcation;
                model_file = optional_model_file(request, lineaging_dir);

                previous_parameters = parameters;
                parameters = legacy_parameters;
                parameter_cleanup = onCleanup( ...
                    @() restore_parameters(previous_parameters));
                load(distribution_file); %#ok<LOAD>
                frame_count = size(movie, 4);
                esequence = cell(1, frame_count);
                detector_cell_diameters = zeros(frame_count, 1);
                detector_effective_cell_counts = zeros(frame_count, 1);
                detector_candidate_diameter_counts = zeros(frame_count, 1);
                previous = 0; %#ok<NASGU>
                for time = 1:frame_count
                    X = single(movie(:, :, :, time)); %#ok<NASGU>
                    nodatause = true; %#ok<NASGU>
                    nodata = true; %#ok<NASGU>
                    usestaticdiameter = use_static_diameter; %#ok<NASGU>
                    firsttimestepdiam = cell_diameter; %#ok<NASGU>
                    downsample = 1; %#ok<NASGU>
                    firsttimestepnumcells = num_cells; %#ok<NASGU>
                    singlevolume = false; %#ok<NASGU>
                    zlevel = size(movie, 3); %#ok<NASGU>
                    ROIpoints = zeros(0, 2); %#ok<NASGU>
                    ROI = false; %#ok<NASGU>
                    ROIxmin = 1; %#ok<NASGU>
                    ROIymin = 1; %#ok<NASGU>
                    savedata = true; %#ok<NASGU>
                    conservememory = false; %#ok<NASGU>
                    e = struct();
                    processVolume;
                    esequence{time} = e;
                    detector_cell_diameters(time) = double(celldiameter);
                    detector_effective_cell_counts(time) = double(numcells);
                    if isfield(e, 'diams')
                        detector_candidate_diameter_counts(time) = numel(e.diams);
                    end
                    previous = e; %#ok<NASGU>
                end

                model_data = load(model_file);
                if ~isfield(model_data, 'trackingparameters') || ...
                        ~isstruct(model_data.trackingparameters) || ...
                        ~isscalar(model_data.trackingparameters)
                    error('ATPy:StarryNiteOracle:InvalidTrackingModel', ...
                        'Tracking model must contain scalar trackingparameters.');
                end
                trackingparameters = model_data.trackingparameters;
                trackingparameters.starttime = 1;
                trackingparameters.endtime = frame_count;
                trackingparameters.trainingmode = false;
                trackingparameters.recordanswers = false;
                trackingparameters.anisotropyvector = [1, 1, anisotropy];
                trackingparameters = apply_tracking_overrides( ...
                    trackingparameters, request);
                parameters.anisotropyvector = [1, 1, anisotropy];
                trainingmode = false; %#ok<NASGU>
                recordanswers = false; %#ok<NASGU>
                skipbifurcation = false; %#ok<NASGU>
                evalforced = false; %#ok<NASGU>
                endtime = frame_count; %#ok<NASGU>

                previous_computed = computedclassificationvector;
                previous_reference = refclassificationvector;
                previous_removed = removed;
                previous_simple_fn = simpleFNcorrect;
                previous_fn_type = FNtype;
                previous_class_round = classround;
                previous_event_trace = ATPY_STARRYNITE_EVENT_TRACE;
                previous_stage_trace = ATPY_STARRYNITE_STAGE_TRACE;
                tracking_classifier_cleanup = onCleanup(@() ...
                    restore_tracking_classifier_globals( ...
                    previous_computed, previous_reference, ...
                    previous_removed, previous_simple_fn, ...
                    previous_fn_type, previous_class_round));
                tracking_event_trace_cleanup = onCleanup(@() ...
                    restore_tracking_event_trace(previous_event_trace));
                tracking_stage_trace_cleanup = onCleanup(@() ...
                    restore_tracking_stage_trace(previous_stage_trace));
                computedclassificationvector = [];
                refclassificationvector = [];
                removed = [];
                simpleFNcorrect = [];
                FNtype = [];
                classround = [];
                ATPY_STARRYNITE_EVENT_TRACE = [];
                ATPY_STARRYNITE_STAGE_TRACE = [];
                tracking_driver_new_classifier_based_version;
                atpy_starrynite_trace('finish', esequence);
                tracking_event_trace = atpy_starrynite_trace('result');
                tracking_stage_trace = atpy_starrynite_stage_trace('result');

                [node_table, edge_table, frame_counts] = ...
                    normalize_tracking_result(esequence);
                classifier_computed_classes = ...
                    double(computedclassificationvector(:));
                classifier_reference_classes = ...
                    double(refclassificationvector(:));
                classifier_removed_diagnostics = double(removed);
                classifier_simple_fn_correct = double(simpleFNcorrect(:));
                classifier_fn_type = double(FNtype);
                classifier_rounds = double(classround(:));
                legacy_node_measurements = ...
                    normalize_legacy_node_measurements(esequence, parameters);
                legacy_self_nn = normalize_legacy_self_nn( ...
                    esequence, trackingparameters);
                extracted_bifurcations = ...
                    extract_retained_bifurcation_features( ...
                        esequence, trackingparameters);
                result.input_shape_yxzt = uint32(size(movie));
                result.upstream_function = upstream_tracking_driver;
                result.detector_function = which('processVolume');
                result.distribution_file = distribution_file;
                result.model_file = model_file;
                result.legacy_parameters = legacy_parameters;
                result.tracking_parameter_summary = ...
                    tracking_parameter_summary(trackingparameters);
                result.node_table = node_table;
                result.node_table_columns = { ...
                    'frame_0based', 'node_0based', ...
                    'x_px_0based', 'y_px_0based', 'z_plane_0based', ...
                    'diameter_xy_px', 'maxima', 'deleted'};
                result.edge_table = edge_table;
                result.edge_table_columns = { ...
                    'source_frame_0based', 'source_node_0based', ...
                    'target_frame_0based', 'target_node_0based', ...
                    'kind_code', 'frame_gap', 'source_deleted'};
                result.edge_kind_codes = { ...
                    '0=link', '1=gap', '2=division'};
                result.frame_detection_counts = frame_counts;
                result.detector_use_static_diameter = use_static_diameter;
                result.detector_cell_diameters_xy = detector_cell_diameters;
                result.detector_effective_cell_counts = ...
                    detector_effective_cell_counts;
                result.detector_candidate_diameter_counts = ...
                    detector_candidate_diameter_counts;
                result.classifier_computed_classes = ...
                    classifier_computed_classes;
                result.classifier_reference_classes = ...
                    classifier_reference_classes;
                result.classifier_removed_diagnostics = ...
                    classifier_removed_diagnostics;
                result.classifier_simple_fn_correct = ...
                    classifier_simple_fn_correct;
                result.classifier_fn_type = classifier_fn_type;
                result.classifier_rounds = classifier_rounds;
                result.tracking_event_trace = tracking_event_trace;
                result.tracking_stage_trace = tracking_stage_trace;
                result.legacy_node_measurements = legacy_node_measurements;
                result.legacy_node_measurement_columns = { ...
                    'frame_0based', 'node_0based', ...
                    'x_px_0based', 'y_px_0based', 'z_plane_0based', ...
                    'diameter_xy_px', 'total_gfp', 'average_gfp', ...
                    'aspect_ratio', 'merged_log_odds_sum', ...
                    'merged_slice_count', 'xy_principal_variance', ...
                    'xy_secondary_variance', ...
                    'self_distance', 'confidence_total_gfp', ...
                    'confidence_nn_z', 'confidence_nn_xy', ...
                    'confidence_aspect_ratio', 'confidence_log_odds', ...
                    'confidence_average_gfp', 'deleted'};
                result.legacy_self_nn = legacy_self_nn;
                result.legacy_self_nn_columns = { ...
                    'frame_0based', 'node_0based', ...
                    'nearest_node_0based', 'mean_self_distance', ...
                    'nearest_z_distance', 'nearest_xy_distance', ...
                    'z_ratio', 'xy_ratio', 'z_log_input', 'xy_log_input', ...
                    'z_log_output', 'xy_log_output'};
                result.extracted_bifurcations = extracted_bifurcations;

            otherwise
                error('ATPy:StarryNiteOracle:UnknownOperation', ...
                    'Unknown request.operation: %s', operation);
        end

        % The legacy tracking driver is a script and assigns a workspace
        % variable named ``result``. Re-attach the oracle envelope after every
        % operation so that script-local names cannot erase provenance.
        result = attach_core_result_metadata( ...
            result, request_path, starrynite_root, operation, distribution_dir);
        if size_from_result(result, operation) < 4
            result.orientation_warning = [ ...
                'The upstream imgaussianAnisotropy function treats arrays with ', ...
                'fewer than four Z planes as color/2-D input. Use Z >= 4 for ', ...
                'three-dimensional parity experiments.'];
        else
            result.orientation_warning = '';
        end
        result.elapsed_seconds = toc(started);
        result.success = true;
        save(result_path, 'result', '-v7');
    catch exception
        result = attach_core_result_metadata( ...
            result, request_path, starrynite_root, '', '');
        result.success = false;
        result.elapsed_seconds = toc(started);
        result.error_identifier = exception.identifier;
        result.error_message = exception.message;
        result.error_report = getReport(exception, 'extended', 'hyperlinks', 'off');
        try
            save(result_path, 'result', '-v7');
        catch save_exception
            warning('ATPy:StarryNiteOracle:ResultSaveFailed', ...
                'Could not save failure result: %s', save_exception.message);
        end
        rethrow(exception);
    end
end


function result = attach_core_result_metadata( ...
        result, request_path, starrynite_root, operation, distribution_dir)
    if ~isstruct(result) || ~isscalar(result)
        result = struct();
    end
    result.schema_version = uint32(1);
    result.request_path = request_path;
    result.starrynite_root = starrynite_root;
    result.matlab_version = version;
    if ~isempty(operation)
        result.operation = operation;
    end
    if ~isempty(distribution_dir)
        result.distribution_code_path = distribution_dir;
        result.image_processing_toolbox_available = ~isempty(ver('images'));
    end
end


function text = require_path_text(value, label)
    if isstring(value) && isscalar(value)
        text = char(value);
    elseif ischar(value) && (isrow(value) || isempty(value))
        text = value;
    else
        error('ATPy:StarryNiteOracle:InvalidArgument', ...
            '%s must be a character vector or scalar string.', label);
    end
    if isempty(strtrim(text))
        error('ATPy:StarryNiteOracle:InvalidArgument', '%s cannot be empty.', label);
    end
end


function path_text = absolute_existing_path(path_text)
    [ok, attributes] = fileattrib(path_text);
    if ~ok
        error('ATPy:StarryNiteOracle:MissingPath', 'Path not found: %s', path_text);
    end
    path_text = attributes.Name;
end


function require_schema_version(request)
    if ~isfield(request, 'schema_version') || ...
            ~isscalar(request.schema_version) || double(request.schema_version) ~= 1
        error('ATPy:StarryNiteOracle:SchemaVersion', ...
            'request.schema_version must be scalar value 1.');
    end
end


function value = require_text_field(request, name)
    if ~isfield(request, name)
        error('ATPy:StarryNiteOracle:MissingField', 'Missing request.%s.', name);
    end
    value = request.(name);
    if isstring(value) && isscalar(value)
        value = char(value);
    elseif ischar(value)
        value = strtrim(value(:)');
    else
        error('ATPy:StarryNiteOracle:InvalidField', ...
            'request.%s must be text.', name);
    end
end


function volume = require_volume(request, name)
    if ~isfield(request, name)
        error('ATPy:StarryNiteOracle:MissingField', 'Missing request.%s.', name);
    end
    volume = request.(name);
    if ~isnumeric(volume) || ~isreal(volume) || ndims(volume) ~= 3 || ...
            any(size(volume) == 0) || any(~isfinite(volume(:)))
        error('ATPy:StarryNiteOracle:InvalidVolume', ...
            'request.%s must be a finite, real, nonempty 3-D numeric array.', name);
    end
end


function movie = require_movie(request, name)
    if ~isfield(request, name)
        error('ATPy:StarryNiteOracle:MissingField', 'Missing request.%s.', name);
    end
    movie = request.(name);
    if ~isnumeric(movie) || ~isreal(movie) || ndims(movie) ~= 4 || ...
            any(size(movie) == 0) || size(movie, 3) < 4 || ...
            size(movie, 4) < 2 || any(~isfinite(movie(:)))
        error('ATPy:StarryNiteOracle:InvalidMovie', ...
            ['request.%s must be a finite, real, nonempty Y-by-X-by-Z-by-T ', ...
             'numeric array with Z >= 4 and T >= 2.'], name);
    end
end


function value = require_axis_vector(request, name, strictly_positive)
    if ~isfield(request, name)
        error('ATPy:StarryNiteOracle:MissingField', 'Missing request.%s.', name);
    end
    value = double(request.(name));
    value = value(:)';
    invalid = numel(value) ~= 3 || any(~isfinite(value));
    if strictly_positive
        invalid = invalid || any(value <= 0);
    else
        invalid = invalid || any(value < 0);
    end
    if invalid
        error('ATPy:StarryNiteOracle:InvalidField', ...
            'request.%s must be a finite three-element %s vector.', ...
            name, ternary(strictly_positive, 'positive', 'nonnegative'));
    end
end


function value = require_finite_vector(request, name, expected_count)
    if ~isfield(request, name)
        error('ATPy:StarryNiteOracle:MissingField', 'Missing request.%s.', name);
    end
    value = double(request.(name));
    value = value(:)';
    invalid = isempty(value) || any(~isfinite(value));
    if expected_count > 0
        invalid = invalid || numel(value) ~= expected_count;
    end
    if invalid
        if expected_count > 0
            requirement = sprintf('%d-element', expected_count);
        else
            requirement = 'nonempty';
        end
        error('ATPy:StarryNiteOracle:InvalidField', ...
            'request.%s must be a finite %s numeric vector.', name, requirement);
    end
end


function value = require_finite_scalar(request, name)
    if ~isfield(request, name)
        error('ATPy:StarryNiteOracle:MissingField', 'Missing request.%s.', name);
    end
    value = double(request.(name));
    if ~isscalar(value) || ~isfinite(value)
        error('ATPy:StarryNiteOracle:InvalidField', ...
            'request.%s must be a finite scalar.', name);
    end
end


function value = require_positive_scalar(request, name)
    value = require_finite_scalar(request, name);
    if value <= 0
        error('ATPy:StarryNiteOracle:InvalidField', ...
            'request.%s must be positive.', name);
    end
end


function value = require_nonnegative_scalar(request, name)
    value = require_finite_scalar(request, name);
    if value < 0
        error('ATPy:StarryNiteOracle:InvalidField', ...
            'request.%s must be nonnegative.', name);
    end
end


function value = optional_scalar(request, name, default)
    if isfield(request, name)
        value = double(request.(name));
        if ~isscalar(value) || ~isfinite(value)
            error('ATPy:StarryNiteOracle:InvalidField', ...
                'request.%s must be a finite scalar.', name);
        end
    else
        value = default;
    end
end


function points = optional_roi_points(request)
    if ~isfield(request, 'roi_points_xy') || isempty(request.roi_points_xy)
        points = zeros(0, 2);
        return;
    end
    points = double(request.roi_points_xy);
    if ~ismatrix(points) || size(points, 2) ~= 2 || any(~isfinite(points(:)))
        error('ATPy:StarryNiteOracle:InvalidField', ...
            'request.roi_points_xy must be an N-by-2 finite numeric array.');
    end
end


function legacy_parameters = require_legacy_parameters(request)
    if ~isfield(request, 'legacy_parameters') || ...
            ~isstruct(request.legacy_parameters) || ...
            ~isscalar(request.legacy_parameters)
        error('ATPy:StarryNiteOracle:InvalidField', ...
            'request.legacy_parameters must be a scalar struct.');
    end
    legacy_parameters = request.legacy_parameters;
    required = { ...
        'staging', ...
        'boundary_percent', ...
        'large_ray_threshold', ...
        'small_ray_threshold'};
    for index = 1:numel(required)
        field_name = required{index};
        if ~isfield(legacy_parameters, field_name)
            error('ATPy:StarryNiteOracle:MissingField', ...
                'request.legacy_parameters.%s is required.', field_name);
        end
        values = legacy_parameters.(field_name);
        if ~isnumeric(values) || ~isreal(values) || isempty(values) || ...
                ~isvector(values) || any(~isfinite(values(:)))
            error('ATPy:StarryNiteOracle:InvalidField', ...
                ['request.legacy_parameters.%s must be a finite, real, ', ...
                 'nonempty numeric vector.'], field_name);
        end
    end
end


function legacy_parameters = require_detector_legacy_parameters(request)
    legacy_parameters = require_legacy_parameters(request);
    required = { ...
        'sigma', ...
        'intensitythreshold', ...
        'rangethreshold', ...
        'nndist_merge', ...
        'mergelower', ...
        'armerge', ...
        'mergesplit', ...
        'split'};
    for index = 1:numel(required)
        field_name = required{index};
        if ~isfield(legacy_parameters, field_name)
            error('ATPy:StarryNiteOracle:MissingField', ...
                'request.legacy_parameters.%s is required.', field_name);
        end
        values = legacy_parameters.(field_name);
        if ~isnumeric(values) || ~isreal(values) || isempty(values) || ...
                ~isvector(values) || any(~isfinite(values(:)))
            error('ATPy:StarryNiteOracle:InvalidField', ...
                ['request.legacy_parameters.%s must be a finite, real, ', ...
                 'nonempty numeric vector.'], field_name);
        end
    end
end


function path_text = optional_distribution_file(request, distribution_dir)
    if isfield(request, 'distribution_file')
        path_text = require_text_field(request, 'distribution_file');
    else
        path_text = fullfile( ...
            distribution_dir, 'clean_distributions_newimage.mat');
    end
    if ~isfile(path_text)
        error('ATPy:StarryNiteOracle:MissingDistributionFile', ...
            'StarryNite distribution file was not found: %s', path_text);
    end
    path_text = absolute_existing_path(path_text);
end


function path_text = optional_model_file(request, lineaging_dir)
    if isfield(request, 'model_file')
        path_text = require_text_field(request, 'model_file');
    else
        path_text = fullfile(lineaging_dir, '2019TrackingModelv2.mat');
    end
    if ~isfile(path_text)
        error('ATPy:StarryNiteOracle:MissingTrackingModel', ...
            'StarryNite tracking model was not found: %s', path_text);
    end
    path_text = absolute_existing_path(path_text);
end


function lineaging_dir = require_lineaging_directory(starrynite_root)
    lineaging_dir = fullfile(starrynite_root, 'distribution_lineaging');
    if ~isfolder(lineaging_dir)
        error('ATPy:StarryNiteOracle:MissingLineagingCode', ...
            'StarryNite distribution_lineaging folder not found: %s', ...
            lineaging_dir);
    end
    lineaging_dir = absolute_existing_path(lineaging_dir);
end


function [trackingparameters, classifier_model, model_file] = ...
        load_tracking_classifier(request, lineaging_dir)
    model_file = optional_model_file(request, lineaging_dir);
    model_data = load(model_file);
    if ~isfield(model_data, 'trackingparameters') || ...
            ~isstruct(model_data.trackingparameters) || ...
            ~isscalar(model_data.trackingparameters)
        error('ATPy:StarryNiteOracle:InvalidTrackingModel', ...
            'Tracking model must contain scalar trackingparameters.');
    end
    trackingparameters = model_data.trackingparameters;
    if ~isfield(trackingparameters, 'bifurcationclassifier') || ...
            ~isstruct(trackingparameters.bifurcationclassifier) || ...
            ~isscalar(trackingparameters.bifurcationclassifier)
        error('ATPy:StarryNiteOracle:InvalidTrackingModel', ...
            ['trackingparameters.bifurcationclassifier must be a scalar ', ...
             'structure.']);
    end
    classifier = trackingparameters.bifurcationclassifier;
    required = {'classifiermodel', 'daughterkeep', 'backkeep', 'forwardkeep'};
    for index = 1:numel(required)
        if ~isfield(classifier, required{index})
            error('ATPy:StarryNiteOracle:InvalidTrackingModel', ...
                'Bifurcation classifier is missing field %s.', required{index});
        end
    end
    classifier_model = classifier.classifiermodel;
    if isempty(classifier_model)
        error('ATPy:StarryNiteOracle:UnavailableClassifierObject', ...
            ['The saved classifier object could not be reconstructed by this ', ...
             'MATLAB release. Legacy NaiveBayes models require a compatible ', ...
             'MATLAB release or a neutral conversion made by one.']);
    end
    if ~(isa(classifier_model, 'ClassificationNaiveBayes') || ...
            isa(classifier_model, 'NaiveBayes'))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierClass', ...
            'Unsupported bifurcation classifier class: %s.', ...
            class(classifier_model));
    end
    daughter_keep = require_classifier_mask(classifier.daughterkeep, 22, ...
        'trackingparameters.bifurcationclassifier.daughterkeep');
    back_keep = require_classifier_mask(classifier.backkeep, 11, ...
        'trackingparameters.bifurcationclassifier.backkeep');
    forward_keep = require_classifier_mask(classifier.forwardkeep, 13, ...
        'trackingparameters.bifurcationclassifier.forwardkeep');
    trackingparameters.bifurcationclassifier.daughterkeep = daughter_keep;
    trackingparameters.bifurcationclassifier.backkeep = back_keep;
    trackingparameters.bifurcationclassifier.forwardkeep = forward_keep;
    selected_count = 1 + sum(daughter_keep) + sum(back_keep) + ...
        sum(forward_keep);
    if isa(classifier_model, 'ClassificationNaiveBayes') && ...
            selected_count ~= numel(classifier_model.DistributionNames)
        error('ATPy:StarryNiteOracle:ClassifierFeatureCount', ...
            ['Feature masks select %d values, but the classifier has %d ', ...
             'predictors.'], selected_count, ...
            numel(classifier_model.DistributionNames));
    end
end


function mask = require_classifier_mask(value, expected_length, label)
    if ~(isnumeric(value) || islogical(value)) || ~isreal(value) || ...
            ~isvector(value) || numel(value) ~= expected_length || ...
            any(~isfinite(double(value(:)))) || ...
            any(~ismember(double(value(:)), [0, 1]))
        error('ATPy:StarryNiteOracle:InvalidTrackingModel', ...
            '%s must be a %d-element logical mask.', label, expected_length);
    end
    mask = logical(value(:));
end


function exported = neutral_classifier_model(trackingparameters, classifier_model)
    if ~isa(classifier_model, 'ClassificationNaiveBayes')
        error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
            ['Neutral export supports ClassificationNaiveBayes models. ', ...
             'Received %s; convert this legacy model with a compatible ', ...
             'MATLAB release first.'], class(classifier_model));
    end
    required_properties = { ...
        'ClassNames', 'Prior', 'Cost', 'DistributionNames', ...
        'DistributionParameters', 'CategoricalLevels', ...
        'PredictorNames', 'CategoricalPredictors', 'NumObservations', ...
        'Kernel', 'Support', 'Width', 'ScoreTransform', 'Mu', 'Sigma'};
    for index = 1:numel(required_properties)
        if ~isprop(classifier_model, required_properties{index})
            error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
                'Classifier is missing required property %s.', ...
                required_properties{index});
        end
    end

    class_names = classifier_model.ClassNames;
    if ~isnumeric(class_names) || ~isreal(class_names) || ...
            ~isvector(class_names) || isempty(class_names) || ...
            any(~isfinite(double(class_names(:))))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
            'Only finite numeric classifier class names can be exported.');
    end
    class_names = double(class_names(:));
    class_count = numel(class_names);
    prior = double(classifier_model.Prior(:)');
    cost = double(classifier_model.Cost);
    if numel(prior) ~= class_count || ...
            ~isequal(size(cost), [class_count, class_count]) || ...
            any(~isfinite(prior)) || any(prior <= 0) || ...
            abs(sum(prior) - 1) > 1e-8 || ...
            any(~isfinite(cost(:))) || any(cost(:) < 0)
        error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
            'Classifier prior or cost dimensions do not match its classes.');
    end

    distribution_names = classifier_model.DistributionNames;
    if ~iscell(distribution_names) || ...
            ~all(cellfun(@(value) ischar(value) && isrow(value), ...
            distribution_names))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
            'DistributionNames must be a cell array of character rows.');
    end
    distribution_names = distribution_names(:)';
    predictor_count = numel(distribution_names);
    score_transform = classifier_model.ScoreTransform;
    if isstring(score_transform) && isscalar(score_transform)
        score_transform = char(score_transform);
    end
    if ~ischar(score_transform) || ~isrow(score_transform) || ...
            ~strcmpi(strtrim(score_transform), 'none')
        error('ATPy:StarryNiteOracle:UnsupportedClassifierScoreTransform', ...
            ['Neutral export requires the identity ScoreTransform ''none''; ', ...
             'received class %s.'], class(classifier_model.ScoreTransform));
    end
    score_transform = 'none';
    [mu, sigma, standardization_state] = ...
        neutral_classifier_standardization(classifier_model, predictor_count);
    distribution_parameters = classifier_model.DistributionParameters;
    if ~iscell(distribution_parameters) || ...
            ~isequal(size(distribution_parameters), ...
            [class_count, predictor_count])
        error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
            'DistributionParameters dimensions do not match the model.');
    end
    categorical_levels = neutral_categorical_levels( ...
        classifier_model.CategoricalLevels, predictor_count);
    kernel_names = neutral_text_cells( ...
        classifier_model.Kernel, predictor_count, 'Kernel');
    support_names = neutral_text_cells( ...
        classifier_model.Support, predictor_count, 'Support');
    widths = double(classifier_model.Width);
    if ~isequal(size(widths), [class_count, predictor_count])
        error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
            'Classifier Width dimensions do not match the model.');
    end

    template = struct( ...
        'class_index_1based', uint32(0), ...
        'predictor_index_1based', uint32(0), ...
        'distribution_name', '', ...
        'numeric_parameters', zeros(0, 1), ...
        'categorical_levels', zeros(0, 1), ...
        'kernel_name', '', ...
        'support', '', ...
        'bandwidth', zeros(0, 1), ...
        'input_data', zeros(0, 1), ...
        'input_frequency', zeros(0, 1), ...
        'input_censored', zeros(0, 1), ...
        'truncation', zeros(0, 1), ...
        'is_truncated', false);
    distributions = repmat(template, class_count, predictor_count);
    for predictor = 1:predictor_count
        distribution_name = lower(strtrim(distribution_names{predictor}));
        if ~ismember(distribution_name, {'mvmn', 'normal', 'kernel'})
            error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
                ['Predictor %d uses unsupported distribution %s. Supported ', ...
                 'neutral exports are mvmn, normal, and kernel.'], ...
                predictor, distribution_names{predictor});
        end
        for class_index = 1:class_count
            item = template;
            item.class_index_1based = uint32(class_index);
            item.predictor_index_1based = uint32(predictor);
            item.distribution_name = distribution_name;
            item.categorical_levels = categorical_levels{predictor};
            item.kernel_name = kernel_names{predictor};
            item.support = support_names{predictor};
            raw = distribution_parameters{class_index, predictor};
            switch distribution_name
                case 'mvmn'
                    item.numeric_parameters = require_neutral_numeric( ...
                        raw, sprintf('mvmn parameter (%d,%d)', ...
                        class_index, predictor), true);
                    if numel(item.numeric_parameters) ~= ...
                            numel(item.categorical_levels)
                        error(['ATPy:StarryNiteOracle:', ...
                            'UnsupportedClassifierDistribution'], ...
                            ['Categorical probabilities at (%d,%d) do not ', ...
                             'match the categorical levels.'], ...
                            class_index, predictor);
                    end
                    if any(item.numeric_parameters < 0) || ...
                            ~isfinite(sum(item.numeric_parameters)) || ...
                            abs(sum(item.numeric_parameters) - 1) > 1e-8
                        error(['ATPy:StarryNiteOracle:', ...
                            'UnsupportedClassifierDistribution'], ...
                            ['Categorical probabilities at (%d,%d) must be ', ...
                             'nonnegative and sum to one.'], ...
                            class_index, predictor);
                    end
                case 'normal'
                    item.numeric_parameters = require_neutral_numeric( ...
                        raw, sprintf('normal parameter (%d,%d)', ...
                        class_index, predictor), true);
                    if numel(item.numeric_parameters) ~= 2 || ...
                            item.numeric_parameters(2) <= 0
                        error(['ATPy:StarryNiteOracle:', ...
                            'UnsupportedClassifierDistribution'], ...
                            ['Normal parameter at (%d,%d) must contain a ', ...
                             'finite mean and positive standard deviation.'], ...
                            class_index, predictor);
                    end
                case 'kernel'
                    item = fill_neutral_kernel_distribution( ...
                        item, raw, class_index, predictor);
            end
            distributions(class_index, predictor) = item;
        end
    end

    predictor_names = classifier_model.PredictorNames;
    valid_predictor_names = iscell(predictor_names) && ...
        all(cellfun(@(value) (ischar(value) && isrow(value)) || ...
        (isstring(value) && isscalar(value)), predictor_names));
    if ~valid_predictor_names || numel(predictor_names) ~= predictor_count
        error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
            'PredictorNames must be one text value per predictor.');
    end
    categorical_predictors = double( ...
        classifier_model.CategoricalPredictors(:)');
    if any(~isfinite(categorical_predictors)) || ...
            any(categorical_predictors ~= fix(categorical_predictors)) || ...
            any(categorical_predictors < 1) || ...
            any(categorical_predictors > predictor_count)
        error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
            'CategoricalPredictors contains invalid predictor indices.');
    end

    classifier = trackingparameters.bifurcationclassifier;
    exported = struct();
    exported.matlab_class = class(classifier_model);
    exported.score_transform = score_transform;
    exported.standardization_state = standardization_state;
    exported.mu = mu;
    exported.sigma = sigma;
    exported.class_names = class_names;
    exported.prior = prior;
    exported.cost = cost;
    exported.predictor_names = predictor_names(:)';
    exported.categorical_predictors_1based = categorical_predictors;
    exported.num_observations = double(classifier_model.NumObservations);
    exported.distribution_names = distribution_names;
    exported.categorical_levels = categorical_levels;
    exported.kernel_names = kernel_names;
    exported.support_names = support_names;
    exported.width = widths;
    exported.distributions = distributions;
    exported.daughter_keep = logical(classifier.daughterkeep(:));
    exported.back_keep = logical(classifier.backkeep(:));
    exported.forward_keep = logical(classifier.forwardkeep(:));
    exported.selected_feature_count = double( ...
        1 + sum(exported.daughter_keep) + sum(exported.back_keep) + ...
        sum(exported.forward_keep));
end


function [mu, sigma, state] = ...
        neutral_classifier_standardization(classifier_model, predictor_count)
    raw_mu = classifier_model.Mu;
    raw_sigma = classifier_model.Sigma;
    if isempty(raw_mu) && isempty(raw_sigma)
        mu = zeros(0, 1);
        sigma = zeros(0, 1);
        state = 'none';
        return;
    end
    if ~isnumeric(raw_mu) || ~isreal(raw_mu) || ~isvector(raw_mu) || ...
            numel(raw_mu) ~= predictor_count || ...
            any(~isfinite(double(raw_mu(:)))) || ...
            ~isnumeric(raw_sigma) || ~isreal(raw_sigma) || ...
            ~isvector(raw_sigma) || numel(raw_sigma) ~= predictor_count || ...
            any(~isfinite(double(raw_sigma(:))))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierStandardization', ...
            ['Classifier Mu and Sigma must both be empty or finite vectors ', ...
             'with one value per predictor.']);
    end
    mu = double(raw_mu(:));
    sigma = double(raw_sigma(:));
    if any(mu ~= 0) || any(sigma ~= 1)
        error('ATPy:StarryNiteOracle:UnsupportedClassifierStandardization', ...
            ['Neutral export cannot reproduce nonidentity predictor ', ...
             'standardization.']);
    end
    state = 'identity';
end


function levels = neutral_categorical_levels(raw_levels, predictor_count)
    if ~iscell(raw_levels) || numel(raw_levels) ~= predictor_count
        error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
            'CategoricalLevels must contain one cell per predictor.');
    end
    levels = cell(1, predictor_count);
    for predictor = 1:predictor_count
        value = raw_levels{predictor};
        if isempty(value)
            levels{predictor} = zeros(0, 1);
        elseif (isnumeric(value) || islogical(value)) && isreal(value) && ...
                isvector(value) && all(isfinite(double(value(:))))
            levels{predictor} = double(value(:));
        else
            error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
                ['Categorical levels for predictor %d are not a finite ', ...
                 'numeric vector.'], predictor);
        end
    end
end


function output = neutral_text_cells(values, expected_count, property_name)
    if ~iscell(values) || numel(values) ~= expected_count
        error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
            'Classifier %s must contain one cell per predictor.', ...
            property_name);
    end
    output = cell(1, expected_count);
    for index = 1:expected_count
        value = values{index};
        if isempty(value)
            output{index} = '';
        elseif ischar(value) && isrow(value)
            output{index} = value;
        elseif isstring(value) && isscalar(value)
            output{index} = char(value);
        else
            error('ATPy:StarryNiteOracle:UnsupportedClassifierExport', ...
                'Classifier %s{%d} is not text or empty.', ...
                property_name, index);
        end
    end
end


function values = require_neutral_numeric(value, label, require_finite)
    if ~(isnumeric(value) || islogical(value)) || ~isreal(value) || ...
            (~isempty(value) && ~isvector(value))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            '%s must be a real numeric vector.', label);
    end
    values = double(value(:));
    if require_finite && any(~isfinite(values))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            '%s must contain finite values.', label);
    end
end


function item = fill_neutral_kernel_distribution( ...
        item, raw, class_index, predictor)
    required_properties = { ...
        'DistributionName', 'Kernel', 'Bandwidth', 'InputData', ...
        'Truncation', 'IsTruncated'};
    if ~isobject(raw)
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            ['Kernel parameter at (%d,%d) has unsupported MATLAB class ', ...
             '%s.'], class_index, predictor, class(raw));
    end
    for index = 1:numel(required_properties)
        if ~isprop(raw, required_properties{index})
            error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
                'Kernel parameter at (%d,%d) is missing property %s.', ...
                class_index, predictor, required_properties{index});
        end
    end
    if ~strcmpi(char(raw.DistributionName), 'Kernel')
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            'Kernel parameter at (%d,%d) reports distribution %s.', ...
            class_index, predictor, char(raw.DistributionName));
    end
    bandwidth = double(raw.Bandwidth);
    if ~isscalar(bandwidth) || ~isfinite(bandwidth) || bandwidth <= 0
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            'Kernel bandwidth at (%d,%d) must be positive and finite.', ...
            class_index, predictor);
    end
    input_data = raw.InputData;
    if ~isstruct(input_data) || ~isscalar(input_data) || ...
            ~all(isfield(input_data, {'data', 'freq', 'cens'}))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            ['Kernel InputData at (%d,%d) must contain data, freq, and ', ...
             'cens fields.'], class_index, predictor);
    end
    data = require_neutral_numeric(input_data.data, ...
        sprintf('kernel InputData.data (%d,%d)', class_index, predictor), true);
    frequency = require_neutral_numeric(input_data.freq, ...
        sprintf('kernel InputData.freq (%d,%d)', class_index, predictor), true);
    censored = require_neutral_numeric(input_data.cens, ...
        sprintf('kernel InputData.cens (%d,%d)', class_index, predictor), false);
    if numel(frequency) ~= numel(data) || ...
            (~isempty(censored) && numel(censored) ~= numel(data))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            'Kernel InputData dimensions disagree at (%d,%d).', ...
            class_index, predictor);
    end
    if any(frequency < 0) || sum(frequency) <= 0 || ...
            (~isempty(censored) && any(~ismember(censored, [0, 1])))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            ['Kernel InputData frequencies/censoring are invalid at ', ...
             '(%d,%d).'], class_index, predictor);
    end
    kernel_name = raw.Kernel;
    if ~(ischar(kernel_name) && isrow(kernel_name))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            'Kernel name at (%d,%d) must be text.', class_index, predictor);
    end
    item.kernel_name = kernel_name;
    item.bandwidth = bandwidth;
    item.input_data = data;
    item.input_frequency = frequency;
    item.input_censored = censored;
    item.truncation = require_neutral_numeric(raw.Truncation, ...
        sprintf('kernel truncation (%d,%d)', class_index, predictor), false);
    if any(isnan(item.truncation))
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            'Kernel truncation contains NaN at (%d,%d).', ...
            class_index, predictor);
    end
    if ~(islogical(raw.IsTruncated) || isnumeric(raw.IsTruncated)) || ...
            ~isscalar(raw.IsTruncated) || ...
            ~ismember(double(raw.IsTruncated), [0, 1])
        error('ATPy:StarryNiteOracle:UnsupportedClassifierDistribution', ...
            'Kernel IsTruncated at (%d,%d) must be scalar logical.', ...
            class_index, predictor);
    end
    item.is_truncated = logical(raw.IsTruncated);
end


function metadata = source_model_metadata(model_file, classifier_model)
    details = dir(model_file);
    if numel(details) ~= 1
        error('ATPy:StarryNiteOracle:InvalidTrackingModel', ...
            'Could not stat tracking model file: %s', model_file);
    end
    metadata = struct();
    metadata.path = model_file;
    metadata.bytes = uint64(details.bytes);
    metadata.modified_datenum = double(details.datenum);
    metadata.trackingparameters_variable = 'trackingparameters';
    metadata.classifier_field = ...
        'trackingparameters.bifurcationclassifier.classifiermodel';
    metadata.matlab_class = class(classifier_model);
end


function row = require_feature_row(request, name, expected_length)
    if ~isfield(request, name)
        error('ATPy:StarryNiteOracle:MissingField', ...
            'Missing request.%s.', name);
    end
    row = request.(name);
    if ~isnumeric(row) || ~isreal(row) || ~ismatrix(row) || ...
            ~isequal(size(row), [1, expected_length]) || any(isinf(row(:)))
        error('ATPy:StarryNiteOracle:InvalidClassifierFeatures', ...
            ['request.%s must be a real 1-by-%d numeric row containing ', ...
             'finite values or NaN.'], name, expected_length);
    end
    row = double(row);
end


function value = require_logical_scalar(request, name)
    if ~isfield(request, name)
        error('ATPy:StarryNiteOracle:MissingField', ...
            'Missing request.%s.', name);
    end
    raw = request.(name);
    if ~(islogical(raw) || isnumeric(raw)) || ~isreal(raw) || ...
            ~isscalar(raw) || ~isfinite(double(raw)) || ...
            ~ismember(double(raw), [0, 1])
        error('ATPy:StarryNiteOracle:InvalidField', ...
            'request.%s must be a scalar logical value.', name);
    end
    value = logical(raw);
end


function value = require_classifier_index(request, name)
    value = require_finite_scalar(request, name);
    if value ~= fix(value) || value == 0 || value < -1
        error('ATPy:StarryNiteOracle:InvalidField', ...
            'request.%s must be -1 or a positive integer.', name);
    end
end


function [classifier_input, topology_class, effective_back, ...
        effective_forward, flags] = assemble_singlemodel_classifier_input( ...
        daughter_data, back_data, forward_data, d1_length, d2_length, ...
        fn_back_1_length, fn_back_2_length, best_forward_d1, ...
        best_forward_d2, trackingparameters)
    if ~isfield(trackingparameters, 'smallcutoff') || ...
            ~isnumeric(trackingparameters.smallcutoff) || ...
            ~isscalar(trackingparameters.smallcutoff) || ...
            ~isfinite(trackingparameters.smallcutoff) || ...
            trackingparameters.smallcutoff <= 0
        error('ATPy:StarryNiteOracle:InvalidTrackingModel', ...
            'trackingparameters.smallcutoff must be positive and finite.');
    end
    small_cutoff = double(trackingparameters.smallcutoff);
    fully_div = d1_length >= small_cutoff && ...
        d2_length >= small_cutoff && ...
        ~(fn_back_1_length > 0 || fn_back_2_length > 0);
    fn_div = d1_length >= small_cutoff && ...
        d2_length >= small_cutoff && ...
        (fn_back_1_length > 0 || fn_back_2_length > 0);
    small = min(d1_length, d2_length) < small_cutoff;
    any_small_lacks_forward = ...
        (d1_length < small_cutoff && ~(best_forward_d1 > 0)) || ...
        (d2_length < small_cutoff && ~(best_forward_d2 > 0));
    has_backward = fn_back_1_length > 0 || fn_back_2_length > 0;
    fully_fp = small && any_small_lacks_forward && ~has_backward;
    dirty_fp = small && any_small_lacks_forward && has_backward;
    div_fp = small && ~any_small_lacks_forward && ~has_backward;
    truly_ambiguous = small && ~any_small_lacks_forward && has_backward;

    topology_values = [ ...
        truly_ambiguous, fully_div, fn_div, fully_fp, dirty_fp || div_fp];
    if sum(topology_values) ~= 1
        error('ATPy:StarryNiteOracle:InvalidClassifierTopology', ...
            'Classifier topology scalars did not select exactly one case.');
    end
    if truly_ambiguous
        topology_class = 1;
    elseif fully_div
        topology_class = 2;
    elseif fn_div
        topology_class = 3;
    elseif fully_fp
        topology_class = 4;
    else
        topology_class = 5;
    end

    effective_back = back_data;
    if div_fp || fully_fp || fully_div
        effective_back(:) = NaN;
    end
    effective_forward = forward_data;
    if fully_fp || div_fp || fn_div || dirty_fp || fully_div
        effective_forward(:) = NaN;
    end
    classifier = trackingparameters.bifurcationclassifier;
    classifier_input = [ ...
        topology_class, ...
        daughter_data(logical(classifier.daughterkeep)), ...
        effective_back(logical(classifier.backkeep)), ...
        effective_forward(logical(classifier.forwardkeep))];

    flags = struct();
    flags.truly_ambiguous = truly_ambiguous;
    flags.fully_division_looking = fully_div;
    flags.false_negative_division_looking = fn_div;
    flags.fully_false_positive_looking = fully_fp;
    flags.dirty_false_positive_looking = dirty_fp;
    flags.division_false_positive_looking = div_fp;
end


function [predicted_class, posterior_scores, class_names] = ...
        direct_classifier_prediction(classifier_model, classifier_input)
    if isa(classifier_model, 'NaiveBayes')
        predicted_class = predict( ...
            classifier_model, classifier_input, 'HandleMissing', 'on');
        posterior_scores = posterior( ...
            classifier_model, classifier_input, 'HandleMissing', 'on');
        if isprop(classifier_model, 'ClassLevels')
            class_names = classifier_model.ClassLevels;
        else
            error('ATPy:StarryNiteOracle:UnsupportedClassifierClass', ...
                'Legacy NaiveBayes model does not expose ClassLevels.');
        end
    else
        [predicted_class, posterior_scores] = predict( ...
            classifier_model, classifier_input);
        class_names = classifier_model.ClassNames;
    end
    if ~isnumeric(predicted_class) || ~isscalar(predicted_class) || ...
            ~isfinite(double(predicted_class)) || ...
            ~isnumeric(posterior_scores) || ~isreal(posterior_scores) || ...
            ~isvector(posterior_scores) || ...
            any(~isfinite(double(posterior_scores(:)))) || ...
            ~isnumeric(class_names) || ~isvector(class_names) || ...
            numel(class_names) ~= numel(posterior_scores)
        error('ATPy:StarryNiteOracle:UnexpectedClassifierOutput', ...
            'Classifier predict/posterior outputs have unexpected shapes.');
    end
    predicted_class = double(predicted_class);
    posterior_scores = double(posterior_scores(:)');
    class_names = double(class_names(:)');
end


function restore_classifier_globals(computed, reference, removed_value, simple_fn)
    global computedclassificationvector; %#ok<GVMIS>
    global refclassificationvector; %#ok<GVMIS>
    global removed; %#ok<GVMIS>
    global simpleFNcorrect; %#ok<GVMIS>
    computedclassificationvector = computed;
    refclassificationvector = reference;
    removed = removed_value;
    simpleFNcorrect = simple_fn;
end


function trace_dir = install_tracking_event_trace(lineaging_dir)
    trace_dir = tempname;
    [created, message] = mkdir(trace_dir);
    if ~created
        error('ATPy:StarryNiteOracle:EventTraceInstallFailed', ...
            'Could not create event-trace instrumentation directory: %s', message);
    end
    try
        source_names = { ...
            'tracking_driver_new_classifier_based_version.m', ...
            'greedydeleteFPbranches.m', 'processOtherBifurcation.m'};
        for index = 1:numel(source_names)
            source_path = fullfile(lineaging_dir, source_names{index});
            if ~isfile(source_path)
                error('ATPy:StarryNiteOracle:EventTraceInstallFailed', ...
                    'Required tracking source was not found: %s', source_path);
            end
            source = fileread(source_path);
            if strcmp(source_names{index}, ...
                    'tracking_driver_new_classifier_based_version.m')
                source = instrument_tracking_driver(source);
            else
                source = strrep(source, ...
                    'predictBifurcationTypeSinglemodel(', ...
                    'atpy_trace_predictBifurcationTypeSinglemodel(');
                source = strrep(source, ...
                    'predictBifurcationType(', ...
                    'atpy_trace_predictBifurcationType(');
            end
            if strcmp(source_names{index}, 'greedydeleteFPbranches.m')
                first_line_end = regexp(source, '\r\n|\n', 'once', 'end');
                if isempty(first_line_end)
                    error('ATPy:StarryNiteOracle:EventTraceInstallFailed', ...
                        'Could not locate the tracking function declaration.');
                end
                source = [ ...
                    source(1:first_line_end), ...
                    'atpy_starrynite_trace(''begin'', esequence);', newline, ...
                    source(first_line_end + 1:end)];
            end
            destination = fullfile(trace_dir, source_names{index});
            write_text_file(destination, source);
        end
    catch exception
        if isfolder(trace_dir)
            rmdir(trace_dir, 's');
        end
        rethrow(exception);
    end
end


function source = instrument_tracking_driver(source)
    initialize_call = [ ...
        '[trackingparameters,esequence]=initializeTrackingStructures', ...
        '(esequence,trackingparameters);'];
    initialize_replacement = [ ...
        'atpy_starrynite_stage_trace(''begin'',esequence,trackingparameters);', ...
        newline, initialize_call, newline, ...
        'atpy_starrynite_stage_trace(''record'',''initialized'',', ...
        'esequence,trackingparameters);'];
    source = replace_required(source, initialize_call, ...
        initialize_replacement, 1, 'initialize tracking');

    easy_call = 'esequence=linkEasyCases(esequence,trackingparameters);';
    easy_replacement = [easy_call, newline, ...
        'atpy_starrynite_stage_trace(''record'',''easy_links'',', ...
        'esequence,trackingparameters);'];
    source = replace_required(source, easy_call, easy_replacement, 1, ...
        'easy links');

    polar_boundary = ...
        '%gather candidates after deleting polar bodies this function is now deleted';
    polar_replacement = [ ...
        'atpy_starrynite_stage_trace(''record'',''post_polar_filter'',', ...
        'esequence,trackingparameters);', newline, polar_boundary];
    source = replace_required(source, polar_boundary, polar_replacement, 1, ...
        'polar-filter boundary');

    candidate_call = 'esequence=gatherEndCandidates(esequence,trackingparameters);';
    candidate_replacement = [candidate_call, newline, ...
        'atpy_starrynite_stage_trace(''record'',''candidates'',', ...
        'esequence,trackingparameters);'];
    source = replace_required(source, candidate_call, candidate_replacement, 1, ...
        'candidate gathering');

    greedy_call = 'esequence=greedyEndScore(esequence,trackingparameters);';
    greedy_replacement = [greedy_call, newline, ...
        'atpy_starrynite_stage_trace(''greedy'',esequence,trackingparameters);'];
    source = replace_required(source, greedy_call, greedy_replacement, 3, ...
        'greedy score stages');

    hysteresis_call = ...
        'esequence=cleanUnlinkedHysteresis(esequence,trackingparameters);';
    hysteresis_replacement = [hysteresis_call, newline, ...
        'atpy_starrynite_stage_trace(''record'',''hysteresis_cleanup'',', ...
        'esequence,trackingparameters);'];
    source = replace_required(source, hysteresis_call, ...
        hysteresis_replacement, 1, 'hysteresis cleanup');

    classifier_call = ...
        'esequence=greedydeleteFPbranches(esequence,trackingparameters);';
    classifier_replacement = [ ...
        'atpy_starrynite_stage_trace(''record'',''geometry_final'',', ...
        'esequence,trackingparameters);', newline, classifier_call];
    source = replace_required(source, classifier_call, ...
        classifier_replacement, 1, 'classifier boundary');
end


function source = replace_required(source, needle, replacement, expected, label)
    matches = strfind(source, needle); %#ok<STREMP>
    if numel(matches) ~= expected
        error('ATPy:StarryNiteOracle:StageTraceInstallFailed', ...
            ['Expected %d occurrence(s) of the %s boundary in the pinned ', ...
             'tracking driver, found %d.'], expected, label, numel(matches));
    end
    source = strrep(source, needle, replacement);
end


function write_text_file(path, value)
    [file_id, message] = fopen(path, 'w');
    if file_id < 0
        error('ATPy:StarryNiteOracle:EventTraceInstallFailed', ...
            'Could not create %s: %s', path, message);
    end
    cleanup = onCleanup(@() fclose(file_id));
    count = fwrite(file_id, value, 'char');
    if count ~= numel(value)
        error('ATPy:StarryNiteOracle:EventTraceInstallFailed', ...
            'Could not completely write %s.', path);
    end
end


function cleanup_tracking_event_trace(trace_dir)
    clear tracking_driver_new_classifier_based_version;
    clear greedydeleteFPbranches processOtherBifurcation;
    current_path = strsplit(path, pathsep);
    if any(strcmpi(current_path, trace_dir))
        rmpath(trace_dir);
    end
    if isfolder(trace_dir)
        rmdir(trace_dir, 's');
    end
end


function restore_tracking_event_trace(value)
    global ATPY_STARRYNITE_EVENT_TRACE; %#ok<GVMIS>
    ATPY_STARRYNITE_EVENT_TRACE = value;
end


function restore_tracking_stage_trace(value)
    global ATPY_STARRYNITE_STAGE_TRACE; %#ok<GVMIS>
    ATPY_STARRYNITE_STAGE_TRACE = value;
end


function restore_tracking_classifier_globals( ...
        computed, reference, removed_value, simple_fn, fn_type, class_round)
    restore_classifier_globals(computed, reference, removed_value, simple_fn);
    global FNtype; %#ok<GVMIS>
    global classround; %#ok<GVMIS>
    FNtype = fn_type;
    classround = class_round;
end


function trackingparameters = apply_tracking_overrides( ...
        trackingparameters, request)
    if ~isfield(request, 'tracking_overrides') || ...
            isempty(request.tracking_overrides)
        return;
    end
    overrides = request.tracking_overrides;
    if ~isstruct(overrides) || ~isscalar(overrides)
        error('ATPy:StarryNiteOracle:InvalidTrackingOverrides', ...
            'request.tracking_overrides must be a scalar struct.');
    end
    allowed = { ...
        'candidateCutoff', 'safefactor', 'nnnumber', 'forwardnnnumber', ...
        'temporalcutoff', 'minnondivscore', 'nondivscorestep', ...
        'maxnondivscore', 'mindivscore', 'divscorestep', 'maxdivscore', ...
        'smallcutoff', 'wideWindow'};
    names = fieldnames(overrides);
    for index = 1:numel(names)
        name = names{index};
        if ~ismember(name, allowed)
            error('ATPy:StarryNiteOracle:InvalidTrackingOverrides', ...
                'Unsupported tracking override: %s', name);
        end
        value = double(overrides.(name));
        if ~isscalar(value) || ~isfinite(value)
            error('ATPy:StarryNiteOracle:InvalidTrackingOverrides', ...
                'Tracking override %s must be a finite scalar.', name);
        end
        trackingparameters.(name) = value;
    end
end


function summary = tracking_parameter_summary(trackingparameters)
    summary = struct();
    names = { ...
        'candidateCutoff', 'safefactor', 'nnnumber', 'forwardnnnumber', ...
        'temporalcutoff', 'temporalcutoffstart', 'interval', 'endtime', ...
        'abscutoff', ...
        'minnondivscore', 'nondivscorestep', ...
        'maxnondivscore', 'mindivscore', 'divscorestep', 'maxdivscore', ...
        'smallcutoff', 'wideWindow'};
    for index = 1:numel(names)
        name = names{index};
        if isfield(trackingparameters, name) && ...
                isnumeric(trackingparameters.(name)) && ...
                isscalar(trackingparameters.(name))
            summary.(name) = double(trackingparameters.(name));
        end
    end
    if isfield(trackingparameters, 'anisotropyvector') && ...
            isnumeric(trackingparameters.anisotropyvector) && ...
            numel(trackingparameters.anisotropyvector) == 3
        summary.anisotropyvector = ...
            double(trackingparameters.anisotropyvector(:)');
    end
end


function restore_parameters(previous_parameters)
    global parameters; %#ok<GVMIS>
    parameters = previous_parameters;
end


function require_upstream_function(function_name, distribution_dir)
    resolved = which(function_name);
    if isempty(resolved)
        error('ATPy:StarryNiteOracle:MissingUpstreamFunction', ...
            'Required StarryNite function is not available: %s', function_name);
    end
    normalized_resolved = normalize_path(resolved);
    normalized_root = [normalize_path(distribution_dir), '/'];
    if ~startsWith(normalized_resolved, normalized_root)
        error('ATPy:StarryNiteOracle:ShadowedUpstreamFunction', ...
            ['%s resolved outside the supplied StarryNite checkout. Resolved ', ...
             'path: %s'], function_name, resolved);
    end
end


function normalized = normalize_path(path_text)
    normalized = strrep(path_text, '\', '/');
    if ispc
        normalized = lower(normalized);
    end
end


function table = as_xyz_table(value, label)
    if isempty(value)
        table = zeros(0, 3);
        return;
    end
    table = double(value);
    if ~ismatrix(table) || size(table, 2) ~= 3
        error('ATPy:StarryNiteOracle:UnexpectedUpstreamOutput', ...
            '%s was expected to be an N-by-3 coordinate array.', label);
    end
end


function converted = xyz_one_to_zyx_zero(coordinates)
    converted = coordinates(:, [3, 2, 1]) - 1;
end


function validate_indices(indices, count)
    if any(~isfinite(indices)) || any(indices ~= fix(indices)) || ...
            any(indices < 1) || any(indices > count)
        error('ATPy:StarryNiteOracle:UnexpectedUpstreamOutput', ...
            'createDiskSet returned invalid center indices.');
    end
end


function selected = select_vector_field(structure, field_name, indices)
    if ~isfield(structure, field_name)
        error('ATPy:StarryNiteOracle:UnexpectedUpstreamOutput', ...
            'createDiskSet output is missing diskSet.%s.', field_name);
    end
    values = double(structure.(field_name));
    values = values(:);
    validate_indices(indices, numel(values));
    selected = values(indices);
end


function result = copy_optional_detection_fields(result, detected)
    coordinate_fields = { ...
        'firstroundpoints', ...
        'sliceCenters'};
    for index = 1:numel(coordinate_fields)
        field_name = coordinate_fields{index};
        if isfield(detected, field_name)
            coordinates = as_xyz_table( ...
                detected.(field_name), ['e.', field_name]);
            result.([field_name, '_xyz_1based']) = coordinates;
            result.([field_name, '_zyx_0based']) = ...
                xyz_one_to_zyx_zero(coordinates);
        end
    end
    numeric_fields = { ...
        'firstroundmaxima', ...
        'diskintensity', ...
        'diskGFPsums', ...
        'diskArea', ...
        'diskMax', ...
        'aspectratio', ...
        'mergedlogoddssum'};
    for index = 1:numel(numeric_fields)
        field_name = numeric_fields{index};
        if isfield(detected, field_name)
            result.(field_name) = double(detected.(field_name));
        end
    end
    if isfield(detected, 'merged_sliceindicies')
        result.merged_slice_indices_1based = detected.merged_sliceindicies;
    end
end


function table = normalize_legacy_node_measurements(esequence, parameters)
    frame_counts = cellfun( ...
        @(item) size(item.finalpoints, 1), esequence);
    table = zeros(sum(frame_counts), 21);
    row = 0;
    for frame = 1:numel(esequence)
        detected = esequence{frame};
        count = size(detected.finalpoints, 1);
        required = { ...
            'finaldiams', 'totalGFP', 'avgGFP', 'aspectratio', ...
            'mergedlogoddssum', 'merged_sliceindicies', ...
            'selfdistance', 'confidencevector'};
        for field_index = 1:numel(required)
            if ~isfield(detected, required{field_index})
                error('ATPy:StarryNiteOracle:MissingFeatureState', ...
                    'esequence{%d}.%s is required for feature extraction.', ...
                    frame, required{field_index});
            end
        end
        if ~isequal(size(detected.confidencevector), [count, 6])
            error('ATPy:StarryNiteOracle:InvalidFeatureState', ...
                'Legacy confidence vectors must be N-by-6.');
        end
        for node = 1:count
            row = row + 1;
            slices = detected.merged_sliceindicies{node};
            xy_variances = legacy_xy_principal_variances( ...
                detected, node, parameters.boundary_percent);
            deleted = 0;
            if isfield(detected, 'delete')
                deleted = double(logical(detected.delete(node)));
            end
            table(row, :) = [ ...
                frame - 1, node - 1, ...
                double(detected.finalpoints(node, :)) - 1, ...
                double(detected.finaldiams(node)), ...
                double(detected.totalGFP(node)), ...
                double(detected.avgGFP(node)), ...
                double(detected.aspectratio(node)), ...
                double(detected.mergedlogoddssum(node)), ...
                numel(slices), xy_variances, ...
                double(detected.selfdistance(node)), ...
                double(detected.confidencevector(node, :)), deleted];
        end
    end
end


function table = normalize_legacy_self_nn(esequence, trackingparameters)
    frame_counts = cellfun( ...
        @(item) size(item.finalpoints, 1), esequence);
    table = zeros(sum(frame_counts), 12);
    row = 0;
    for frame = 1:numel(esequence)
        points = esequence{frame}.finalpoints;
        count = size(points, 1);
        if count == 0
            continue;
        end
        distances = distance_anisotropic( ...
            points', points', trackingparameters.anisotropyvector);
        for node = 1:count
            distances(node, node) = Inf;
        end
        [~, nearest] = min(distances);
        distances_z = distance(points(:, 3)', points(:, 3)') .* ...
            trackingparameters.anisotropyvector(3);
        distances_xy = distance(points(:, 1:2)', points(:, 1:2)');
        mean_self_distance = mean(esequence{frame}.selfdistance);
        for node = 1:count
            row = row + 1;
            z_distance = distances_z(node, nearest(node));
            xy_distance = distances_xy(node, nearest(node));
            z_ratio = z_distance ./ mean_self_distance;
            xy_ratio = xy_distance ./ mean_self_distance;
            z_log_input = z_ratio + 1;
            xy_log_input = xy_ratio + 1;
            table(row, :) = [ ...
                frame - 1, node - 1, nearest(node) - 1, ...
                mean_self_distance, z_distance, xy_distance, ...
                z_ratio, xy_ratio, z_log_input, xy_log_input, ...
                log(z_log_input), log(xy_log_input)];
        end
    end
end


function values = legacy_xy_principal_variances( ...
        detected, node, boundary_percent)
    slices = detected.merged_sliceindicies{node};
    if isempty(slices)
        values = [0, 0];
        return;
    end
    maxima = detected.diskintensity(slices);
    slices = slices(maxima >= max(maxima) * boundary_percent);
    positions = zeros(0, 2);
    for slice_index = 1:numel(slices)
        current = slices(slice_index);
        nonzero = find( ...
            detected.xpositions(:, current) ~= 0 | ...
            detected.ypositions(:, current) ~= 0);
        positions = [positions; ...
            double(detected.xpositions(nonzero, current)), ...
            double(detected.ypositions(nonzero, current))]; %#ok<AGROW>
    end
    if numel(slices) == 1
        positions = [positions; positions];
    end
    if size(positions, 1) < 2
        values = [0, 0];
        return;
    end
    [~, ~, latent] = pca(positions);
    if isempty(latent)
        values = [0, 0];
    else
        values = zeros(1, 2);
        values(1:min(2, numel(latent))) = ...
            double(latent(1:min(2, numel(latent))));
    end
end


function extracted = extract_retained_bifurcation_features( ...
        esequence, trackingparameters)
    extracted = struct();
    extracted.node_references = zeros(0, 6);
    extracted.daughter_data = zeros(0, 22);
    extracted.backward_data = zeros(0, 11);
    extracted.forward_data = zeros(0, 13);
    extracted.topology = zeros(0, 9);
    extracted.nondivision_scores = zeros(0, 2);
    extracted.division_scores = zeros(0, 6);
    extracted.selected_backward_candidate = zeros(0, 3);
    extracted.backward_matching = zeros(0, 3);
    extracted.backward_players_start = zeros(0, 3);
    extracted.backward_players_end = zeros(0, 3);
    extracted.daughter1_backward_candidates = cell(0, 1);
    extracted.daughter2_backward_candidates = cell(0, 1);
    extracted.daughter_columns = arrayfun( ...
        @(index) sprintf('daughter_%02d', index), 1:22, ...
        'UniformOutput', false);
    extracted.backward_columns = arrayfun( ...
        @(index) sprintf('backward_%02d', index), 1:11, ...
        'UniformOutput', false);
    extracted.forward_columns = arrayfun( ...
        @(index) sprintf('forward_%02d', index), 1:13, ...
        'UniformOutput', false);

    for frame = 1:numel(esequence)
        detected = esequence{frame};
        if ~isfield(detected, 'suc')
            continue;
        end
        for node = 1:size(detected.suc, 1)
            if detected.delete(node) || any(detected.suc(node, :) <= 0)
                continue;
            end
            initialize_bifurcation_feature_globals();
            count = 1;
            [d1cand, d2cand, d1candt, d2candt, d1length, d2length, ...
                best_back_correct, best_matching, back1_length, ...
                back2_length, best_forward1, best_forward2, ...
                players_start, players_end, best_daughter, best_index, ...
                division_scores, daughter_data, forward_data, ...
                backward_data, ~] = assembleBifurcationData( ...
                    esequence, frame, node, trackingparameters, count);
            daughter1 = detected.suc(node, 1);
            daughter2 = detected.suc(node, 2);
            daughter1_time = detected.suc_time(node, 1);
            daughter2_time = detected.suc_time(node, 2);
            nondivision_scores = nondivScoreModelCostFunction( ...
                esequence, node, frame, detected.suc(node, :)', ...
                detected.suc_time(node, :)', trackingparameters);

            extracted.node_references(end + 1, :) = [ ...
                frame - 1, node - 1, daughter1_time - 1, daughter1 - 1, ...
                daughter2_time - 1, daughter2 - 1];
            extracted.daughter_data(end + 1, :) = double(daughter_data);
            extracted.backward_data(end + 1, :) = double(backward_data);
            extracted.forward_data(end + 1, :) = double(forward_data);
            extracted.topology(end + 1, :) = [ ...
                d1length, d2length, back1_length, back2_length, ...
                best_forward1, best_forward2, best_back_correct, ...
                best_daughter, best_index];
            extracted.nondivision_scores(end + 1, :) = ...
                double(nondivision_scores(:)');
            extracted.division_scores(end + 1, :) = ...
                double(division_scores(:)');
            if best_daughter == 1
                selected = [1, d1candt(best_index) - 1, ...
                    d1cand(best_index) - 1];
            elseif best_daughter == 2
                selected = [2, d2candt(best_index) - 1, ...
                    d2cand(best_index) - 1];
            else
                selected = [-1, -1, -1];
            end
            extracted.selected_backward_candidate(end + 1, :) = selected;
            extracted.backward_matching(end + 1, :) = ...
                legacy_three_vector(best_matching);
            extracted.backward_players_start(end + 1, :) = ...
                legacy_three_vector(players_start);
            extracted.backward_players_end(end + 1, :) = ...
                legacy_three_vector(players_end);
            extracted.daughter1_backward_candidates{end + 1, 1} = ...
                [double(d1candt(:)) - 1, double(d1cand(:)) - 1];
            extracted.daughter2_backward_candidates{end + 1, 1} = ...
                [double(d2candt(:)) - 1, double(d2cand(:)) - 1];
        end
    end
end


function result = legacy_three_vector(value)
    values = double(value(:)');
    if isscalar(values)
        result = [values, 0, 0];
    elseif numel(values) == 3
        result = values;
    else
        error('ATPy:StarryNiteOracle:InvalidRepairDiagnostics', ...
            'Repair matching diagnostics must contain one or three values.');
    end
end


function initialize_bifurcation_feature_globals()
    global BifurcationMeasures; %#ok<GVMIS>
    global confidenceData; %#ok<GVMIS>
    global splitFNMatchScore; %#ok<GVMIS>
    BifurcationMeasures = [];
    confidenceData = struct( ...
        'bifcon', [], 'bifconvector', [], 'bestbackconv', [], ...
        'bestforward1conv', [], 'bestforward2conv', [], ...
        'rforwardd1_conv', [], 'rforwardd1_consum', [], ...
        'rforwardd1_flength', [], 'rforwardd1_solidlength', [], ...
        'rforwardd2_conv', [], 'rforwardd2_consum', [], ...
        'rforwardd2_flength', [], 'rforwardd2_solidlength', []);
    splitFNMatchScore = struct( ...
        'forwardd1xy', [], 'forwardd2xy', [], ...
        'forwardd1z', [], 'forwardd2z', [], ...
        'backxy', [], 'backz', [], ...
        'forwardd1gapsize', [], 'forwardd2gapsize', [], ...
        'backgapsize', []);
end


function [nodes, edges, frame_counts] = normalize_tracking_result(esequence)
    frame_count = numel(esequence);
    frame_counts = zeros(frame_count, 1);
    for frame = 1:frame_count
        frame_counts(frame) = size(esequence{frame}.finalpoints, 1);
    end
    total_nodes = sum(frame_counts);
    nodes = zeros(total_nodes, 8);
    edges = zeros(2 * total_nodes, 7);
    node_offset = 0;
    edge_count = 0;
    for frame = 1:frame_count
        detected = esequence{frame};
        points = as_xyz_table(detected.finalpoints, 'esequence.finalpoints');
        count = size(points, 1);
        diameters = double(detected.finaldiams(:));
        maxima = double(detected.finalmaximas(:));
        if numel(diameters) ~= count || numel(maxima) ~= count
            error('ATPy:StarryNiteOracle:UnexpectedTrackingOutput', ...
                'Tracking node measurements do not match finalpoints.');
        end
        deleted = zeros(count, 1);
        if isfield(detected, 'delete')
            deleted = double(logical(detected.delete(:)));
        end
        if numel(deleted) ~= count
            error('ATPy:StarryNiteOracle:UnexpectedTrackingOutput', ...
                'Tracking delete flags do not match finalpoints.');
        end
        local_nodes = [ ...
            repmat(frame - 1, count, 1), ...
            (0:count - 1)', ...
            points - 1, ...
            diameters, maxima, deleted];
        node_indices = node_offset + (1:count);
        nodes(node_indices, :) = local_nodes;
        node_offset = node_offset + count;

        if ~isfield(detected, 'suc') || ~isfield(detected, 'suc_time')
            continue;
        end
        successors = double(detected.suc);
        successor_times = double(detected.suc_time);
        if ~isequal(size(successors), [count, 2]) || ...
                ~isequal(size(successor_times), [count, 2])
            error('ATPy:StarryNiteOracle:UnexpectedTrackingOutput', ...
                'Tracking successor tables must be N-by-2.');
        end
        for node = 1:count
            is_division = all(successors(node, :) > 0);
            for slot = 1:2
                target = successors(node, slot);
                target_time = successor_times(node, slot);
                if target <= 0
                    continue;
                end
                if target_time <= frame || target_time > frame_count
                    error('ATPy:StarryNiteOracle:UnexpectedTrackingOutput', ...
                        'Tracking returned an invalid successor time.');
                end
                target_count = size(esequence{target_time}.finalpoints, 1);
                if target ~= fix(target) || target > target_count
                    error('ATPy:StarryNiteOracle:UnexpectedTrackingOutput', ...
                        'Tracking returned an invalid successor index.');
                end
                if is_division
                    kind = 2;
                elseif target_time > frame + 1
                    kind = 1;
                else
                    kind = 0;
                end
                edge_count = edge_count + 1;
                edges(edge_count, :) = [ ...
                    frame - 1, node - 1, target_time - 1, target - 1, ...
                    kind, target_time - frame, deleted(node)];
            end
        end
    end
    edges = edges(1:edge_count, :);
end


function z_size = size_from_result(result, operation)
    if strcmp(operation, 'slice_candidates') || isfield(result, 'input_shape_yxz')
        shape = double(result.input_shape_yxz);
        if numel(shape) >= 3
            z_size = shape(3);
            return;
        end
    end
    z_size = inf;
end


function result = ternary(condition, true_value, false_value)
    if condition
        result = true_value;
    else
        result = false_value;
    end
end
