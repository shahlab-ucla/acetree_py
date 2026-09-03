function export_starrynite_classifier_numeric(request_path, result_path)
%EXPORT_STARRYNITE_CLASSIFIER_NUMERIC Export old NaiveBayes state safely.
%
% This helper is intentionally limited to syntax and APIs available in old
% MATLAB releases that can still reconstruct the pre-R2014b NaiveBayes class.
% It exchanges v7 MAT files, never writes into the source model, and emits
% only numeric, logical, character, cell, and scalar-structure values.

    result = struct();
    result.success = false;
    result.export_schema = 'acetree.starrynite-matlab-classifier-export';
    result.export_version = uint32(1);
    result.matlab_version = version;
    try
        result.matlab_release = version('-release');
    catch
        result.matlab_release = '';
    end
    started = tic;

    try
        request_path = require_text(request_path, 'request_path');
        result_path = require_text(result_path, 'result_path');
        loaded_request = load(request_path);
        if ~isfield(loaded_request, 'request') || ...
                ~isstruct(loaded_request.request) || ...
                numel(loaded_request.request) ~= 1
            error('ATPy:StarryNiteExport:InvalidRequest', ...
                'Request MAT file must contain one scalar request structure.');
        end
        request = loaded_request.request;
        require_request_schema(request);
        model_file = require_request_text(request, 'model_file');
        expected_kind = lower(require_request_text(request, 'expected_kind'));
        if ~any(strcmp(expected_kind, {'auto', 'single', 'ambigious'}))
            error('ATPy:StarryNiteExport:InvalidRequest', ...
                'expected_kind must be auto, single, or ambigious.');
        end
        if exist(model_file, 'file') ~= 2
            error('ATPy:StarryNiteExport:MissingModel', ...
                'Tracking model file was not found: %s', model_file);
        end
        model_data = load(model_file);
        if ~isfield(model_data, 'trackingparameters') || ...
                ~isstruct(model_data.trackingparameters) || ...
                numel(model_data.trackingparameters) ~= 1
            error('ATPy:StarryNiteExport:InvalidTrackingModel', ...
                'Model must contain scalar trackingparameters.');
        end
        trackingparameters = model_data.trackingparameters;
        if ~isfield(trackingparameters, 'bifurcationclassifier') || ...
                ~isstruct(trackingparameters.bifurcationclassifier) || ...
                numel(trackingparameters.bifurcationclassifier) ~= 1
            error('ATPy:StarryNiteExport:InvalidTrackingModel', ...
                ['trackingparameters.bifurcationclassifier must be a ', ...
                 'scalar structure.']);
        end
        classifier = trackingparameters.bifurcationclassifier;
        daughter_keep = require_mask(classifier, 'daughterkeep', 22);
        back_keep = require_mask(classifier, 'backkeep', 11);
        forward_keep = require_mask(classifier, 'forwardkeep', 13);

        family_names = {'ambigious', 'fp_div', 'dirtyfp_fn', 'divfp'};
        family_present = false(1, numel(family_names));
        for index = 1:numel(family_names)
            family_present(index) = isfield(classifier, family_names{index});
        end
        has_single = isfield(classifier, 'classifiermodel');
        if any(family_present) && ~all(family_present)
            error('ATPy:StarryNiteExport:PartialClassifierFamily', ...
                ['Historical classifier family is incomplete; all of ', ...
                 'ambigious, fp_div, dirtyfp_fn, and divfp are required.']);
        end
        if has_single && all(family_present)
            error('ATPy:StarryNiteExport:AmbiguousClassifierLayout', ...
                ['Tracking model contains both classifiermodel and the four ', ...
                 'historical family fields. Refusing to guess which is active.']);
        end
        if ~has_single && ~all(family_present)
            error('ATPy:StarryNiteExport:MissingClassifier', ...
                'Tracking model contains no supported bifurcation classifier.');
        end

        result.source_model = source_metadata(model_file);
        if has_single
            model_kind = 'single';
        else
            model_kind = 'ambigious';
        end
        if ~strcmp(expected_kind, 'auto') && ~strcmp(expected_kind, model_kind)
            error('ATPy:StarryNiteExport:ClassifierKindMismatch', ...
                'Model contains %s classifier data, not requested %s.', ...
                model_kind, expected_kind);
        end
        result.model_kind = model_kind;

        if has_single
            model = classifier.classifiermodel;
            require_available_classifier(model, 'classifiermodel');
            result.classifier_matlab_classes = {class(model)};
            if isa(model, 'ClassificationNaiveBayes')
                % The main oracle has a separately validated modern exporter.
                result.export_complete = false;
                result.requires_current_oracle = true;
            elseif isa(model, 'NaiveBayes')
                expected_predictors = 1 + sum(daughter_keep) + ...
                    sum(back_keep) + sum(forward_keep);
                [result.classifier_model, result.validation_probes] = ...
                    legacy_single_model( ...
                    model, daughter_keep, back_keep, forward_keep, ...
                    expected_predictors);
                result.export_complete = true;
                result.requires_current_oracle = false;
            else
                error('ATPy:StarryNiteExport:UnsupportedClassifierClass', ...
                    'Unsupported classifiermodel MATLAB class: %s.', class(model));
            end
        else
            models = struct();
            validation_probes = struct();
            classes = cell(1, numel(family_names));
            expected_labels = { [0, 2, 3], [0, 1, 3], ...
                [0, 1, 2, 3], [0, 1, 3] };
            expected_predictors = [ ...
                sum(daughter_keep) + sum(back_keep) + sum(forward_keep), ...
                sum(daughter_keep), ...
                sum(daughter_keep) + sum(back_keep), ...
                sum(daughter_keep) + sum(forward_keep) ];
            for index = 1:numel(family_names)
                name = family_names{index};
                model = classifier.(name);
                require_available_classifier(model, name);
                classes{index} = class(model);
                if ~isa(model, 'NaiveBayes')
                    error('ATPy:StarryNiteExport:UnsupportedClassifierClass', ...
                        ['Historical family member %s has MATLAB class %s. ', ...
                         'Only reconstructable NaiveBayes family objects are ', ...
                         'accepted by the compatibility exporter.'], ...
                        name, class(model));
                end
                [models.(name), validation_probes.(name)] = ...
                    legacy_family_submodel( ...
                    model, expected_predictors(index), name, ...
                    expected_labels{index});
            end
            family = struct();
            family.family_name = 'ambigious';
            family.daughter_keep = daughter_keep;
            family.back_keep = back_keep;
            family.forward_keep = forward_keep;
            family.submodels = models;
            result.classifier_family_model = family;
            result.validation_probes = validation_probes;
            result.classifier_matlab_classes = classes;
            result.export_complete = true;
            result.requires_current_oracle = false;
        end
        result.elapsed_seconds = toc(started);
        result.success = true;
        save(result_path, 'result', '-v7');
    catch exception
        result.elapsed_seconds = toc(started);
        result.error_identifier = exception.identifier;
        result.error_message = exception.message;
        try
            save(result_path, 'result', '-v7');
        catch
        end
        rethrow(exception);
    end
end


function require_request_schema(request)
    required = {'schema_version', 'operation', 'model_file', 'expected_kind'};
    for index = 1:numel(required)
        if ~isfield(request, required{index})
            error('ATPy:StarryNiteExport:InvalidRequest', ...
                'Request is missing field %s.', required{index});
        end
    end
    version_value = double(request.schema_version);
    if ~isscalar(version_value) || ~isfinite(version_value) || version_value ~= 1
        error('ATPy:StarryNiteExport:InvalidRequest', ...
            'Only request schema_version 1 is supported.');
    end
    operation = require_request_text(request, 'operation');
    if ~strcmp(operation, 'export_classifier_numeric_compatible')
        error('ATPy:StarryNiteExport:InvalidRequest', ...
            'Unsupported request operation: %s', operation);
    end
end


function value = require_request_text(request, name)
    if ~isfield(request, name)
        error('ATPy:StarryNiteExport:InvalidRequest', ...
            'Request is missing field %s.', name);
    end
    value = require_text(request.(name), ['request.' name]);
end


function value = require_text(value, label)
    if ~(ischar(value) && (isrow(value) || isempty(value)))
        error('ATPy:StarryNiteExport:InvalidText', ...
            '%s must be a character row.', label);
    end
end


function mask = require_mask(classifier, name, expected_length)
    if ~isfield(classifier, name)
        error('ATPy:StarryNiteExport:InvalidTrackingModel', ...
            'Bifurcation classifier is missing field %s.', name);
    end
    value = classifier.(name);
    if ~(isnumeric(value) || islogical(value)) || ~isreal(value) || ...
            ~isvector(value) || numel(value) ~= expected_length || ...
            any(~isfinite(double(value(:)))) || ...
            any(~ismember(double(value(:)), [0, 1]))
        error('ATPy:StarryNiteExport:InvalidTrackingModel', ...
            '%s must be a %d-element logical mask.', name, expected_length);
    end
    mask = logical(value(:));
end


function require_available_classifier(model, name)
    if isempty(model)
        error('ATPy:StarryNiteExport:UnavailableClassifierObject', ...
            ['Classifier %s reconstructed as empty in MATLAB %s. Select an ', ...
             'older MATLAB release that still supports the saved NaiveBayes ', ...
             'class; no approximate substitute will be exported.'], ...
            name, version);
    end
    if numel(model) ~= 1
        error('ATPy:StarryNiteExport:InvalidClassifierObject', ...
            'Classifier %s must be a scalar object.', name);
    end
end


function metadata = source_metadata(model_file)
    details = dir(model_file);
    if numel(details) ~= 1
        error('ATPy:StarryNiteExport:InvalidTrackingModel', ...
            'Could not stat tracking model: %s', model_file);
    end
    metadata = struct();
    metadata.path = model_file;
    metadata.bytes = double(details.bytes);
    metadata.modified_datenum = double(details.datenum);
    metadata.trackingparameters_variable = 'trackingparameters';
    metadata.classifier_field = 'trackingparameters.bifurcationclassifier';
end


function [exported, probes] = legacy_single_model(model, daughter_keep, back_keep, ...
        forward_keep, expected_predictors)
    components = legacy_components(model, expected_predictors);
    if ~isequal(components.class_labels(:)', [0, 1, 2, 3])
        error('ATPy:StarryNiteExport:UnexpectedClassOrder', ...
            ['Single-model ClassLevels must be ordered [0 1 2 3]; ', ...
             'received a different order.']);
    end
    [entries, categorical_levels, kernel_names, support_names, widths] = ...
        raw_distribution_entries(model, components);
    names = feature_names(expected_predictors, 'single');
    exported = struct();
    exported.matlab_class = 'NaiveBayes';
    exported.score_transform = 'none';
    exported.standardization_state = 'none';
    exported.mu = zeros(0, 1);
    exported.sigma = zeros(0, 1);
    exported.class_names = components.class_labels;
    exported.prior = components.prior;
    exported.cost = default_cost(numel(components.class_labels));
    exported.predictor_names = names;
    exported.categorical_predictors_1based = find(strcmp( ...
        components.distribution_names, 'mvmn'));
    exported.num_observations = legacy_observation_count(model);
    exported.distribution_names = components.distribution_names;
    exported.categorical_levels = categorical_levels;
    exported.kernel_names = kernel_names;
    exported.support_names = support_names;
    exported.width = widths;
    exported.distributions = entries;
    exported.daughter_keep = daughter_keep;
    exported.back_keep = back_keep;
    exported.forward_keep = forward_keep;
    exported.selected_feature_count = double(expected_predictors);
    probes = legacy_validation_probes(model, components);
end


function [exported, probes] = legacy_family_submodel(model, expected_predictors, name, ...
        expected_labels)
    components = legacy_components(model, expected_predictors);
    if ~isequal(components.class_labels(:)', expected_labels)
        error('ATPy:StarryNiteExport:UnexpectedClassOrder', ...
            ['Family member %s ClassLevels do not match StarryNite''s ', ...
             'required positional class order.'], name);
    end
    distributions = family_distributions(model, components);
    exported = struct();
    exported.classifier_family = 'legacy_classifier';
    exported.feature_names = feature_names(expected_predictors, name);
    exported.class_labels = components.class_labels;
    exported.class_priors = components.prior;
    exported.misclassification_costs = ...
        default_cost(numel(components.class_labels));
    exported.distributions = distributions;
    exported.missing_value_policy = 'omit';
    exported.tie_policy = 'first_class';
    probes = legacy_validation_probes(model, components);
end


function components = legacy_components(model, expected_predictors)
    class_labels = model_property(model, 'ClassLevels');
    if ~isnumeric(class_labels) || ~isreal(class_labels) || ...
            ~isvector(class_labels) || isempty(class_labels) || ...
            any(~isfinite(double(class_labels(:)))) || ...
            any(double(class_labels(:)) ~= fix(double(class_labels(:))))
        error('ATPy:StarryNiteExport:UnsupportedLegacyModel', ...
            'NaiveBayes ClassLevels must be finite numeric integers.');
    end
    class_labels = double(class_labels(:)');
    prior = model_property(model, 'Prior');
    if ~isnumeric(prior) || ~isreal(prior) || ~isvector(prior) || ...
            numel(prior) ~= numel(class_labels) || ...
            any(~isfinite(double(prior(:)))) || any(double(prior(:)) <= 0)
        error('ATPy:StarryNiteExport:UnsupportedLegacyModel', ...
            'NaiveBayes Prior must contain one positive finite value per class.');
    end
    prior = double(prior(:)');
    if abs(sum(prior) - 1) > 1e-8
        error('ATPy:StarryNiteExport:UnsupportedLegacyModel', ...
            'NaiveBayes Prior must sum to one.');
    end
    distribution_names = normalize_distribution_names( ...
        model_property(model, 'Dist'), expected_predictors);
    parameters = model_property(model, 'Params');
    if ~iscell(parameters)
        error('ATPy:StarryNiteExport:UnsupportedLegacyModel', ...
            'NaiveBayes Params must be a class-by-predictor cell array.');
    end
    class_count = numel(class_labels);
    if isequal(size(parameters), [expected_predictors, class_count]) && ...
            ~isequal(size(parameters), [class_count, expected_predictors])
        parameters = parameters';
    end
    if ~isequal(size(parameters), [class_count, expected_predictors])
        error('ATPy:StarryNiteExport:ClassifierFeatureCount', ...
            ['NaiveBayes Params has shape %d-by-%d; expected %d classes by ', ...
             '%d mask-selected predictors.'], size(parameters, 1), ...
            size(parameters, 2), class_count, expected_predictors);
    end
    components = struct();
    components.class_labels = class_labels;
    components.prior = prior;
    components.distribution_names = distribution_names;
    components.parameters = parameters;
    components.class_count = class_count;
    components.predictor_count = expected_predictors;
end


function names = normalize_distribution_names(raw, expected_count)
    if ischar(raw) && isrow(raw)
        raw_names = repmat({raw}, 1, expected_count);
    elseif iscell(raw) && numel(raw) == expected_count
        raw_names = raw(:)';
    else
        error('ATPy:StarryNiteExport:UnsupportedLegacyModel', ...
            'NaiveBayes Dist must name one distribution per predictor.');
    end
    names = cell(1, expected_count);
    for predictor = 1:expected_count
        value = raw_names{predictor};
        if ~(ischar(value) && isrow(value))
            error('ATPy:StarryNiteExport:UnsupportedLegacyModel', ...
                'NaiveBayes Dist{%d} is not text.', predictor);
        end
        name = lower(strtrim(value));
        if strcmp(name, 'gaussian')
            name = 'normal';
        end
        if ~any(strcmp(name, {'normal', 'mvmn', 'kernel'}))
            error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
                'NaiveBayes predictor %d uses unsupported distribution %s.', ...
                predictor, value);
        end
        names{predictor} = name;
    end
end


function [entries, categorical_levels, kernel_names, support_names, widths] = ...
        raw_distribution_entries(model, components)
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
    entries = repmat(template, components.class_count, ...
        components.predictor_count);
    categorical_levels = cell(1, components.predictor_count);
    kernel_names = repmat({''}, 1, components.predictor_count);
    support_names = repmat({''}, 1, components.predictor_count);
    widths = NaN(components.class_count, components.predictor_count);
    for predictor = 1:components.predictor_count
        distribution_name = components.distribution_names{predictor};
        if strcmp(distribution_name, 'mvmn')
            categories = legacy_categories(model, predictor);
            categorical_levels{predictor} = categories;
        else
            categorical_levels{predictor} = zeros(0, 1);
        end
        for class_index = 1:components.class_count
            item = template;
            item.class_index_1based = uint32(class_index);
            item.predictor_index_1based = uint32(predictor);
            item.distribution_name = distribution_name;
            raw = components.parameters{class_index, predictor};
            if strcmp(distribution_name, 'normal')
                item.numeric_parameters = normal_parameters( ...
                    raw, class_index, predictor);
            elseif strcmp(distribution_name, 'mvmn')
                probabilities = probability_vector(raw, class_index, predictor);
                if numel(probabilities) ~= numel(categorical_levels{predictor})
                    error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
                        ['mvmn probabilities at (%d,%d) do not match ', ...
                         'UniqVal categories.'], class_index, predictor);
                end
                item.numeric_parameters = probabilities;
                item.categorical_levels = categorical_levels{predictor};
            else
                kernel = legacy_kernel(model, raw, class_index, predictor);
                item.kernel_name = kernel.kernel_name;
                item.support = kernel.support;
                item.bandwidth = kernel.bandwidth;
                item.input_data = kernel.samples;
                item.input_frequency = kernel.frequencies;
                item.input_censored = zeros(0, 1);
                item.truncation = [-Inf; Inf];
                item.is_truncated = false;
                widths(class_index, predictor) = kernel.bandwidth;
                if isempty(kernel_names{predictor})
                    kernel_names{predictor} = kernel.kernel_name;
                    support_names{predictor} = kernel.support;
                elseif ~strcmp(kernel_names{predictor}, kernel.kernel_name) || ...
                        ~strcmp(support_names{predictor}, kernel.support)
                    error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
                        ['Kernel type/support varies by class for predictor ', ...
                         '%d.'], predictor);
                end
            end
            entries(class_index, predictor) = item;
        end
    end
end


function distributions = family_distributions(model, components)
    distributions = cell(1, components.predictor_count);
    for predictor = 1:components.predictor_count
        name = components.distribution_names{predictor};
        if strcmp(name, 'normal')
            means = zeros(1, components.class_count);
            deviations = zeros(1, components.class_count);
            for class_index = 1:components.class_count
                values = normal_parameters( ...
                    components.parameters{class_index, predictor}, ...
                    class_index, predictor);
                means(class_index) = values(1);
                deviations(class_index) = values(2);
            end
            distribution = struct();
            distribution.kind = 'gaussian';
            distribution.means = means;
            distribution.standard_deviations = deviations;
        elseif strcmp(name, 'mvmn')
            categories = legacy_categories(model, predictor);
            probabilities = zeros(components.class_count, numel(categories));
            for class_index = 1:components.class_count
                row = probability_vector( ...
                    components.parameters{class_index, predictor}, ...
                    class_index, predictor);
                if numel(row) ~= numel(categories)
                    error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
                        ['mvmn probabilities at (%d,%d) do not match ', ...
                         'UniqVal categories.'], class_index, predictor);
                end
                probabilities(class_index, :) = row(:)';
            end
            distribution = struct();
            distribution.kind = 'categorical';
            distribution.categories = categories(:)';
            distribution.probabilities = probabilities;
        else
            samples = cell(1, components.class_count);
            frequencies = cell(1, components.class_count);
            bandwidths = zeros(1, components.class_count);
            kernel_name = '';
            support = '';
            for class_index = 1:components.class_count
                kernel = legacy_kernel(model, ...
                    components.parameters{class_index, predictor}, ...
                    class_index, predictor);
                if isempty(kernel_name)
                    kernel_name = kernel.kernel_name;
                    support = kernel.support;
                elseif ~strcmp(kernel_name, kernel.kernel_name) || ...
                        ~strcmp(support, kernel.support)
                    error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
                        ['Kernel type/support varies by class for predictor ', ...
                         '%d.'], predictor);
                end
                samples{class_index} = kernel.samples;
                frequencies{class_index} = kernel.frequencies;
                bandwidths(class_index) = kernel.bandwidth;
            end
            distribution = struct();
            distribution.kind = 'kernel';
            distribution.kernel = kernel_name;
            distribution.support = support;
            distribution.samples = samples;
            distribution.frequencies = frequencies;
            distribution.bandwidths = bandwidths;
        end
        distributions{predictor} = distribution;
    end
end


function probes = legacy_validation_probes(model, components)
    % A center/mode row and a scale/category perturbation per class exercise
    % every decoded parameter column, then the original object supplies the
    % authoritative outputs.
    features = zeros(2 * components.class_count, components.predictor_count);
    for target_class = 1:components.class_count
        offset_row = components.class_count + target_class;
        for predictor = 1:components.predictor_count
            name = components.distribution_names{predictor};
            raw = components.parameters{target_class, predictor};
            if strcmp(name, 'normal')
                values = normal_parameters(raw, target_class, predictor);
                features(target_class, predictor) = values(1);
                features(offset_row, predictor) = values(1) + 0.5 * values(2);
            elseif strcmp(name, 'mvmn')
                categories = legacy_categories(model, predictor);
                probabilities = probability_vector( ...
                    raw, target_class, predictor);
                [unused, category_index] = max(probabilities); %#ok<ASGLU>
                features(target_class, predictor) = categories(category_index);
                alternate_index = 1 + mod(category_index, numel(categories));
                features(offset_row, predictor) = categories(alternate_index);
            else
                kernel = legacy_kernel( ...
                    model, raw, target_class, predictor);
                [unused, sample_index] = max(kernel.frequencies); %#ok<ASGLU>
                features(target_class, predictor) = kernel.samples(sample_index);
                features(offset_row, predictor) = ...
                    kernel.samples(sample_index) + 0.5 * kernel.bandwidth;
            end
        end
    end
    if any(~isfinite(features(:)))
        error('ATPy:StarryNiteExport:InvalidValidationProbe', ...
            'Classifier center/mode validation probes must be finite.');
    end
    try
        predicted = predict(model, features, 'HandleMissing', 'on');
        posterior_scores = posterior( ...
            model, features, 'HandleMissing', 'on');
    catch exception
        error('ATPy:StarryNiteExport:ValidationProbeFailed', ...
            ['Could not evaluate validation probes through the loaded ', ...
             'NaiveBayes object: %s'], exception.message);
    end
    if ~isnumeric(predicted) || ~isreal(predicted) || ...
            ~isvector(predicted) || numel(predicted) ~= size(features, 1) || ...
            any(~isfinite(double(predicted(:)))) || ...
            ~isnumeric(posterior_scores) || ~isreal(posterior_scores) || ...
            ~isequal(size(posterior_scores), ...
            [size(features, 1), components.class_count]) || ...
            any(~isfinite(double(posterior_scores(:)))) || ...
            any(double(posterior_scores(:)) < 0)
        error('ATPy:StarryNiteExport:InvalidValidationProbe', ...
            'NaiveBayes validation probe outputs have invalid shapes or values.');
    end
    posterior_scores = double(posterior_scores);
    for row = 1:size(posterior_scores, 1)
        if abs(sum(posterior_scores(row, :)) - 1) > 1e-7
            error('ATPy:StarryNiteExport:InvalidValidationProbe', ...
                'NaiveBayes validation posterior row %d does not sum to one.', row);
        end
    end
    probes = struct();
    probes.features = features;
    probes.predicted_classes = double(predicted(:));
    probes.posteriors = posterior_scores;
    probes.class_labels = components.class_labels;
end


function values = normal_parameters(raw, class_index, predictor)
    if ~isnumeric(raw) || ~isreal(raw) || ~isvector(raw) || ...
            numel(raw) ~= 2 || any(~isfinite(double(raw(:))))
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            ['Normal Params{%d,%d} must contain finite mean and standard ', ...
             'deviation.'], class_index, predictor);
    end
    values = double(raw(:));
    if values(2) <= 0
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'Normal standard deviation at (%d,%d) must be positive.', ...
            class_index, predictor);
    end
end


function values = probability_vector(raw, class_index, predictor)
    if ~isnumeric(raw) || ~isreal(raw) || ~isvector(raw) || isempty(raw) || ...
            any(~isfinite(double(raw(:)))) || any(double(raw(:)) < 0)
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'mvmn Params{%d,%d} must be finite probabilities.', ...
            class_index, predictor);
    end
    values = double(raw(:));
    if abs(sum(values) - 1) > 1e-8
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'mvmn Params{%d,%d} must sum to one.', class_index, predictor);
    end
end


function categories = legacy_categories(model, predictor)
    raw = model_property(model, 'UniqVal');
    if ~iscell(raw) || numel(raw) < predictor
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'NaiveBayes UniqVal is missing predictor %d.', predictor);
    end
    value = raw{predictor};
    if ~isnumeric(value) || ~isreal(value) || ~isvector(value) || ...
            isempty(value) || any(~isfinite(double(value(:)))) || ...
            numel(unique(double(value(:)))) ~= numel(value)
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'NaiveBayes UniqVal{%d} must contain unique finite values.', predictor);
    end
    categories = double(value(:));
end


function kernel = legacy_kernel(model, raw, class_index, predictor)
    input_data = optional_object_value(raw, 'InputData');
    if isempty(input_data) && isnumeric(raw) && isreal(raw) && isvector(raw)
        samples = double(raw(:));
        frequencies = ones(size(samples));
        censored = zeros(0, 1);
    elseif isstruct(input_data) && isscalar(input_data) && ...
            isfield(input_data, 'data')
        samples = require_finite_vector(input_data.data, ...
            'kernel InputData.data', false);
        if isfield(input_data, 'freq') && ~isempty(input_data.freq)
            frequencies = require_finite_vector(input_data.freq, ...
                'kernel InputData.freq', true);
        else
            frequencies = ones(size(samples));
        end
        if isfield(input_data, 'cens')
            censored = double(input_data.cens(:));
        else
            censored = zeros(0, 1);
        end
    else
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            ['Kernel Params{%d,%d} does not expose inert InputData or raw ', ...
             'numeric samples.'], class_index, predictor);
    end
    if isempty(samples) || numel(frequencies) ~= numel(samples) || ...
            any(frequencies < 0) || sum(frequencies) <= 0
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'Kernel samples/frequencies are invalid at (%d,%d).', ...
            class_index, predictor);
    end
    if ~isempty(censored) && (numel(censored) ~= numel(samples) || ...
            any(~ismember(censored, [0, 1])) || any(censored ~= 0))
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'Censored kernel data is unsupported at (%d,%d).', ...
            class_index, predictor);
    end

    bandwidth = optional_object_value(raw, 'Bandwidth');
    if isempty(bandwidth)
        bandwidth = indexed_model_setting(model, 'KernelWidth', ...
            class_index, predictor);
    end
    if ~isnumeric(bandwidth) || ~isreal(bandwidth) || ...
            ~isscalar(bandwidth) || ~isfinite(double(bandwidth)) || bandwidth <= 0
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'Kernel bandwidth must be positive at (%d,%d).', ...
            class_index, predictor);
    end

    kernel_name = optional_object_value(raw, 'Kernel');
    if isempty(kernel_name)
        kernel_name = indexed_model_setting(model, 'KernelType', ...
            class_index, predictor);
    end
    kernel_name = normalize_kernel_text(kernel_name, 'kernel type');
    if strcmp(kernel_name, 'gaussian')
        kernel_name = 'normal';
    end
    if ~strcmp(kernel_name, 'normal')
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'Only normal/gaussian legacy kernels are supported; received %s.', ...
            kernel_name);
    end

    support = optional_object_value(raw, 'Support');
    if isempty(support)
        support = indexed_model_setting(model, 'KernelSupport', ...
            class_index, predictor);
    end
    support = normalize_kernel_text(support, 'kernel support');
    if ~any(strcmp(support, {'unbounded', 'unbounded-support'}))
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'Only unbounded legacy kernels are supported; received %s.', support);
    end

    is_truncated = optional_object_value(raw, 'IsTruncated');
    if ~isempty(is_truncated) && ...
            (~isscalar(is_truncated) || logical(is_truncated))
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            'Truncated kernel distribution is unsupported at (%d,%d).', ...
            class_index, predictor);
    end
    kernel = struct();
    kernel.samples = samples;
    kernel.frequencies = frequencies;
    kernel.bandwidth = double(bandwidth);
    kernel.kernel_name = kernel_name;
    kernel.support = 'unbounded';
end


function values = require_finite_vector(value, label, nonnegative)
    if ~isnumeric(value) || ~isreal(value) || ~isvector(value) || ...
            any(~isfinite(double(value(:))))
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            '%s must be a finite numeric vector.', label);
    end
    values = double(value(:));
    if nonnegative && any(values < 0)
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            '%s cannot contain negative values.', label);
    end
end


function value = optional_object_value(object, name)
    value = [];
    try
        if isobject(object) && isprop(object, name)
            value = object.(name);
            return;
        end
    catch
    end
    try
        structure = struct(object);
        if isfield(structure, name)
            value = structure.(name);
        end
    catch
    end
end


function value = model_property(model, name)
    try
        if isprop(model, name)
            value = model.(name);
            return;
        end
    catch
    end
    try
        structure = struct(model);
    catch
        error('ATPy:StarryNiteExport:UnsupportedLegacyModel', ...
            'Could not inspect required NaiveBayes property %s.', name);
    end
    if ~isfield(structure, name)
        error('ATPy:StarryNiteExport:UnsupportedLegacyModel', ...
            'NaiveBayes object is missing required property %s.', name);
    end
    value = structure.(name);
end


function value = indexed_model_setting(model, name, class_index, predictor)
    raw = model_property(model, name);
    if ischar(raw)
        value = raw;
    elseif iscell(raw)
        if isequal(size(raw), [1, 1])
            value = raw{1};
        elseif size(raw, 1) >= class_index && size(raw, 2) >= predictor
            value = raw{class_index, predictor};
        elseif numel(raw) >= predictor
            value = raw{predictor};
        else
            value = [];
        end
    elseif isnumeric(raw)
        if isscalar(raw)
            value = raw;
        elseif size(raw, 1) >= class_index && size(raw, 2) >= predictor
            value = raw(class_index, predictor);
        elseif numel(raw) >= predictor
            value = raw(predictor);
        else
            value = [];
        end
    else
        value = [];
    end
end


function value = normalize_kernel_text(value, label)
    if ~(ischar(value) && isrow(value))
        error('ATPy:StarryNiteExport:UnsupportedDistribution', ...
            '%s must be text.', label);
    end
    value = lower(strtrim(value));
end


function count = legacy_observation_count(model)
    count = 0;
    try
        sizes = model_property(model, 'ClassSize');
        if isnumeric(sizes) && isreal(sizes) && isvector(sizes) && ...
                all(isfinite(double(sizes(:)))) && all(double(sizes(:)) >= 0)
            count = sum(double(sizes(:)));
        end
    catch
        count = 0;
    end
end


function names = feature_names(count, prefix)
    names = cell(1, count);
    for index = 1:count
        names{index} = sprintf('%s_feature_%03d', prefix, index);
    end
end


function costs = default_cost(class_count)
    costs = ones(class_count, class_count) - eye(class_count);
end
