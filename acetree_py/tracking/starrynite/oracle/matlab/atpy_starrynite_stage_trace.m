function output = atpy_starrynite_stage_trace(action, varargin)
%ATPY_STARRYNITE_STAGE_TRACE Capture geometry-linking stage boundaries.
%
% The live oracle instruments a temporary copy of StarryNite's whole-movie
% driver.  This helper records raw predecessor/successor arrays after each
% meaningful geometry stage, together with the candidate relation and the
% active greedy threshold.  The supplied StarryNite checkout is never
% modified.

    global ATPY_STARRYNITE_STAGE_TRACE; %#ok<GVMIS>

    if isstring(action) && isscalar(action)
        action = char(action);
    end
    if ~ischar(action)
        error('ATPy:StarryNiteOracle:InvalidStageTraceAction', ...
            'Stage-trace action must be text.');
    end

    switch action
        case 'begin'
            require_argument_count(action, varargin, 2);
            trace = empty_trace();
            ATPY_STARRYNITE_STAGE_TRACE = append_stage( ...
                trace, 'detected', varargin{1}, varargin{2}, NaN);
            output = [];

        case 'record'
            require_argument_count(action, varargin, 3);
            require_started_trace();
            label = require_label(varargin{1});
            ATPY_STARRYNITE_STAGE_TRACE = append_stage( ...
                ATPY_STARRYNITE_STAGE_TRACE, label, varargin{2}, ...
                varargin{3}, NaN);
            output = [];

        case 'greedy'
            require_argument_count(action, varargin, 2);
            require_started_trace();
            trackingparameters = varargin{2};
            if isfield(trackingparameters, 'trackdiv') && ...
                    logical(trackingparameters.trackdiv)
                label = 'division';
                threshold = scalar_parameter( ...
                    trackingparameters, 'endscorethresh_div');
            elseif isfield(trackingparameters, 'tracknondiv') && ...
                    logical(trackingparameters.tracknondiv)
                label = 'nondivision';
                threshold = scalar_parameter( ...
                    trackingparameters, 'endscorethresh_nondiv');
            else
                error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
                    'A greedy stage must select division or nondivision mode.');
            end
            ATPY_STARRYNITE_STAGE_TRACE = append_stage( ...
                ATPY_STARRYNITE_STAGE_TRACE, label, varargin{1}, ...
                trackingparameters, threshold);
            output = [];

        case 'result'
            require_argument_count(action, varargin, 0);
            require_started_trace();
            if isempty(ATPY_STARRYNITE_STAGE_TRACE.stage_labels) || ...
                    ~strcmp( ...
                        ATPY_STARRYNITE_STAGE_TRACE.stage_labels{end}, ...
                        'geometry_final')
                error('ATPy:StarryNiteOracle:UnfinishedStageTrace', ...
                    'Stage trace did not reach the geometry_final boundary.');
            end
            ATPY_STARRYNITE_STAGE_TRACE.finished = true;
            output = ATPY_STARRYNITE_STAGE_TRACE;

        otherwise
            error('ATPy:StarryNiteOracle:InvalidStageTraceAction', ...
                'Unknown stage-trace action: %s', action);
    end
end


function trace = empty_trace()
    trace = struct();
    trace.schema_version = uint32(1);
    trace.snapshot_columns = { ...
        'frame_0based', 'node_0based', 'deleted', ...
        'predecessor_frame_0based', 'predecessor_node_0based', ...
        'successor1_frame_0based', 'successor1_node_0based', ...
        'successor2_frame_0based', 'successor2_node_0based'};
    trace.candidate_columns = { ...
        'direction_code', ...
        'source_frame_0based', 'source_node_0based', ...
        'target_frame_0based', 'target_node_0based'};
    trace.stage_labels = cell(0, 1);
    trace.stage_thresholds = zeros(0, 1);
    trace.stage_snapshots = cell(0, 1);
    trace.stage_candidate_tables = cell(0, 1);
    trace.stage_node_counts = zeros(0, 1);
    trace.stage_active_counts = zeros(0, 1);
    trace.stage_deleted_counts = zeros(0, 1);
    trace.stage_edge_counts = zeros(0, 1);
    trace.stage_candidate_counts = zeros(0, 1);
    trace.stage_count = 0;
    trace.finished = false;
end


function trace = append_stage(trace, label, esequence, trackingparameters, threshold)
    if trace.finished
        error('ATPy:StarryNiteOracle:FinishedStageTrace', ...
            'Cannot append to a finalized stage trace.');
    end
    validate_tracking_parameters(trackingparameters);
    snapshot = lineage_pointer_snapshot(esequence);
    candidates = forward_candidate_snapshot(esequence);
    trace.stage_labels{end + 1, 1} = label;
    trace.stage_thresholds(end + 1, 1) = double(threshold);
    trace.stage_snapshots{end + 1, 1} = snapshot;
    trace.stage_candidate_tables{end + 1, 1} = candidates;
    trace.stage_node_counts(end + 1, 1) = size(snapshot, 1);
    trace.stage_deleted_counts(end + 1, 1) = sum(snapshot(:, 3) ~= 0);
    trace.stage_active_counts(end + 1, 1) = ...
        size(snapshot, 1) - trace.stage_deleted_counts(end, 1);
    trace.stage_edge_counts(end + 1, 1) = ...
        sum(snapshot(:, 6) >= 0) + sum(snapshot(:, 8) >= 0);
    trace.stage_candidate_counts(end + 1, 1) = size(candidates, 1);
    trace.stage_count = double(numel(trace.stage_labels));
end


function table = lineage_pointer_snapshot(esequence)
    require_esequence(esequence);
    counts = frame_counts(esequence);
    table = zeros(sum(counts), 9);
    output_row = 0;
    for time = 1:numel(esequence)
        detected = esequence{time};
        count = counts(time);
        [deleted, pred, pred_time, suc, suc_time] = ...
            normalized_pointer_arrays(detected, count);
        for node = 1:count
            output_row = output_row + 1;
            predecessor = normalized_reference(pred_time(node), pred(node));
            successor1 = normalized_reference(suc_time(node, 1), suc(node, 1));
            successor2 = normalized_reference(suc_time(node, 2), suc(node, 2));
            table(output_row, :) = [ ...
                time - 1, node - 1, double(logical(deleted(node))), ...
                predecessor, successor1, successor2];
        end
    end
end


function table = forward_candidate_snapshot(esequence)
    require_esequence(esequence);
    rows = cell(0, 1);
    count = 0;
    for time = 1:numel(esequence)
        detected = esequence{time};
        if ~isfield(detected, 'forwardcandidates')
            continue;
        end
        if numel(detected.forwardcandidates) ~= size(detected.finalpoints, 1)
            error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
                'forwardcandidates does not match the detection count.');
        end
        for node = 1:numel(detected.forwardcandidates)
            candidates = detected.forwardcandidates{node};
            if isempty(candidates)
                continue;
            end
            candidates = double(candidates);
            if ~ismatrix(candidates) || size(candidates, 2) ~= 2 || ...
                    any(~isfinite(candidates), 'all') || ...
                    any(candidates ~= fix(candidates), 'all') || ...
                    any(candidates <= 0, 'all')
                error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
                    'forwardcandidates must contain positive integer node/time pairs.');
            end
            for index = 1:size(candidates, 1)
                count = count + 1;
                rows{count, 1} = [ ... %#ok<AGROW>
                    0, time - 1, node - 1, ...
                    candidates(index, 2) - 1, candidates(index, 1) - 1];
            end
        end
    end
    for time = 1:numel(esequence)
        detected = esequence{time};
        if ~isfield(detected, 'backcandidates')
            continue;
        end
        if numel(detected.backcandidates) ~= size(detected.finalpoints, 1)
            error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
                'backcandidates does not match the detection count.');
        end
        for node = 1:numel(detected.backcandidates)
            candidates = detected.backcandidates{node};
            if isempty(candidates)
                continue;
            end
            candidates = double(candidates);
            if ~ismatrix(candidates) || size(candidates, 2) ~= 2 || ...
                    any(~isfinite(candidates), 'all') || ...
                    any(candidates ~= fix(candidates), 'all') || ...
                    any(candidates <= 0, 'all')
                error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
                    'backcandidates must contain positive integer node/time pairs.');
            end
            for index = 1:size(candidates, 1)
                count = count + 1;
                rows{count, 1} = [ ... %#ok<AGROW>
                    1, candidates(index, 2) - 1, ...
                    candidates(index, 1) - 1, time - 1, node - 1];
            end
        end
    end
    if count == 0
        table = zeros(0, 5);
    else
        table = vertcat(rows{:});
    end
end


function [deleted, pred, pred_time, suc, suc_time] = ...
        normalized_pointer_arrays(detected, count)
    deleted = zeros(count, 1);
    pred = -ones(count, 1);
    pred_time = -ones(count, 1);
    suc = -ones(count, 2);
    suc_time = -ones(count, 2);
    if isfield(detected, 'delete')
        deleted = double(detected.delete(:));
    end
    if isfield(detected, 'pred')
        pred = double(detected.pred(:));
    end
    if isfield(detected, 'pred_time')
        pred_time = double(detected.pred_time(:));
    end
    if isfield(detected, 'suc')
        suc = double(detected.suc);
    end
    if isfield(detected, 'suc_time')
        suc_time = double(detected.suc_time);
    end
    if numel(deleted) ~= count || numel(pred) ~= count || ...
            numel(pred_time) ~= count || ~isequal(size(suc), [count, 2]) || ...
            ~isequal(size(suc_time), [count, 2])
        error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
            'Lineage pointer arrays do not match the detection count.');
    end
end


function counts = frame_counts(esequence)
    counts = zeros(numel(esequence), 1);
    for time = 1:numel(esequence)
        detected = esequence{time};
        if ~isstruct(detected) || ~isscalar(detected) || ...
                ~isfield(detected, 'finalpoints') || ...
                ~ismatrix(detected.finalpoints) || ...
                size(detected.finalpoints, 2) ~= 3
            error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
                'Each frame must contain an N-by-3 finalpoints array.');
        end
        counts(time) = size(detected.finalpoints, 1);
    end
end


function require_esequence(esequence)
    if ~iscell(esequence) || isempty(esequence)
        error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
            'Geometry stage state must be a non-empty cell array.');
    end
end


function reference = normalized_reference(time, node)
    time = double(time);
    node = double(node);
    if time <= 0 || node <= 0
        if time ~= -1 || node ~= -1
            error('ATPy:StarryNiteOracle:InvalidStageTraceReference', ...
                'Absent lineage references must use the -1/-1 sentinel.');
        end
        reference = [-1, -1];
        return;
    end
    if time ~= fix(time) || node ~= fix(node)
        error('ATPy:StarryNiteOracle:InvalidStageTraceReference', ...
            'Lineage references must be integer indices.');
    end
    reference = [time - 1, node - 1];
end


function value = scalar_parameter(trackingparameters, name)
    if ~isfield(trackingparameters, name)
        error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
            'trackingparameters.%s is required for stage tracing.', name);
    end
    value = double(trackingparameters.(name));
    if ~isscalar(value) || isnan(value)
        error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
            'trackingparameters.%s must be a scalar.', name);
    end
end


function validate_tracking_parameters(trackingparameters)
    if ~isstruct(trackingparameters) || ~isscalar(trackingparameters)
        error('ATPy:StarryNiteOracle:InvalidStageTraceState', ...
            'trackingparameters must be a scalar structure.');
    end
end


function label = require_label(value)
    if isstring(value) && isscalar(value)
        value = char(value);
    end
    if ~ischar(value) || isempty(value)
        error('ATPy:StarryNiteOracle:InvalidStageTraceLabel', ...
            'Stage label must be non-empty text.');
    end
    label = value;
end


function require_started_trace()
    global ATPY_STARRYNITE_STAGE_TRACE; %#ok<GVMIS>
    if isempty(ATPY_STARRYNITE_STAGE_TRACE) || ...
            ~isstruct(ATPY_STARRYNITE_STAGE_TRACE) || ...
            ~isfield(ATPY_STARRYNITE_STAGE_TRACE, 'schema_version')
        error('ATPy:StarryNiteOracle:MissingStageTrace', ...
            'Stage trace was used before initialization.');
    end
end


function require_argument_count(action, values, expected)
    if numel(values) ~= expected
        error('ATPy:StarryNiteOracle:InvalidStageTraceArguments', ...
            'Stage-trace action %s expects %d argument(s).', action, expected);
    end
end
