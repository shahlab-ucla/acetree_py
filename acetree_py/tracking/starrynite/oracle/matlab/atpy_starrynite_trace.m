function output = atpy_starrynite_trace(action, varargin)
%ATPY_STARRYNITE_TRACE Record classifier-boundary lineage pointer snapshots.
%
% The oracle instruments temporary copies of greedydeleteFPbranches.m and
% processOtherBifurcation.m.  The upstream checkout is never modified.  One
% snapshot is retained at phase entry, immediately before every classifier
% invocation, and after the phase finishes.  Python derives deterministic
% mutation batches by diffing adjacent snapshots.

    global ATPY_STARRYNITE_EVENT_TRACE; %#ok<GVMIS>

    if isstring(action) && isscalar(action)
        action = char(action);
    end
    if ~ischar(action)
        error('ATPy:StarryNiteOracle:InvalidEventTraceAction', ...
            'Event trace action must be text.');
    end

    switch action
        case 'begin'
            require_argument_count(action, varargin, 1);
            trace = struct();
            trace.schema_version = uint32(1);
            trace.classification_table = zeros(0, 14);
            trace.classification_columns = { ...
                'event_index_0based', 'checkpoint_index_0based', ...
                'parent_frame_0based', 'parent_node_0based', ...
                'daughter1_frame_0based', 'daughter1_node_0based', ...
                'daughter2_frame_0based', 'daughter2_node_0based', ...
                'classifier_round', 'computed_class', ...
                'effective_class', 'predicted_class', ...
                'force_mode', 'classifier_family_code'};
            trace.classifier_family_codes = { ...
                '0=single_model', '1=ambigious_multi_model'};
            trace.snapshot_columns = { ...
                'frame_0based', 'node_0based', 'deleted', ...
                'predecessor_frame_0based', 'predecessor_node_0based', ...
                'successor1_frame_0based', 'successor1_node_0based', ...
                'successor2_frame_0based', 'successor2_node_0based'};
            trace.snapshots = cell(0, 1);
            trace.snapshots{end + 1, 1} = lineage_pointer_snapshot(varargin{1});
            trace.snapshot_count = double(numel(trace.snapshots));
            trace.finished = false;
            ATPY_STARRYNITE_EVENT_TRACE = trace;
            output = [];

        case 'classification'
            require_argument_count(action, varargin, 9);
            require_started_trace();
            if ATPY_STARRYNITE_EVENT_TRACE.finished
                error('ATPy:StarryNiteOracle:FinishedEventTrace', ...
                    'Cannot append a classification after trace finalization.');
            end
            esequence = varargin{1};
            time = require_positive_integer(varargin{2}, 'classification time');
            parent = require_positive_integer(varargin{3}, 'classification parent');
            classifier_round = require_positive_integer( ...
                varargin{4}, 'classifier round');
            if ~ismember(classifier_round, [1, 2])
                error('ATPy:StarryNiteOracle:InvalidEventTraceRound', ...
                    'Classifier round must be 1 or 2.');
            end
            predicted_class = require_class(varargin{5}, 'predicted class', false);
            computed_class = require_class(varargin{6}, 'computed class', true);
            effective_class = require_class(varargin{7}, 'effective class', false);
            force_mode = require_logical_scalar(varargin{8}, 'force mode');
            classifier_family = require_nonnegative_integer( ...
                varargin{9}, 'classifier family code');
            if ~ismember(classifier_family, [0, 1])
                error('ATPy:StarryNiteOracle:InvalidEventTraceClassifier', ...
                    'Classifier family code must be 0 or 1.');
            end
            if predicted_class ~= effective_class
                error('ATPy:StarryNiteOracle:EventTraceClassMismatch', ...
                    'Predicted and effective classifier values disagree.');
            end
            if time > numel(esequence) || ...
                    parent > size(esequence{time}.finalpoints, 1)
                error('ATPy:StarryNiteOracle:InvalidEventTraceNode', ...
                    'Classification parent is outside the movie state.');
            end
            successors = double(esequence{time}.suc(parent, :));
            successor_times = double(esequence{time}.suc_time(parent, :));
            if any(successors <= 0) || any(successor_times <= 0)
                error('ATPy:StarryNiteOracle:InvalidEventTraceBifurcation', ...
                    'A recorded classification must have two successors.');
            end
            checkpoint_index = numel(ATPY_STARRYNITE_EVENT_TRACE.snapshots);
            ATPY_STARRYNITE_EVENT_TRACE.snapshots{end + 1, 1} = ...
                lineage_pointer_snapshot(esequence);
            event_index = size( ...
                ATPY_STARRYNITE_EVENT_TRACE.classification_table, 1);
            ATPY_STARRYNITE_EVENT_TRACE.classification_table(end + 1, :) = [ ...
                event_index, checkpoint_index, time - 1, parent - 1, ...
                successor_times(1) - 1, successors(1) - 1, ...
                successor_times(2) - 1, successors(2) - 1, ...
                classifier_round, computed_class, effective_class, ...
                predicted_class, double(force_mode), classifier_family];
            ATPY_STARRYNITE_EVENT_TRACE.snapshot_count = ...
                double(numel(ATPY_STARRYNITE_EVENT_TRACE.snapshots));
            output = [];

        case 'finish'
            require_argument_count(action, varargin, 1);
            require_started_trace();
            if ATPY_STARRYNITE_EVENT_TRACE.finished
                error('ATPy:StarryNiteOracle:FinishedEventTrace', ...
                    'Event trace was finalized more than once.');
            end
            ATPY_STARRYNITE_EVENT_TRACE.snapshots{end + 1, 1} = ...
                lineage_pointer_snapshot(varargin{1});
            ATPY_STARRYNITE_EVENT_TRACE.snapshot_count = ...
                double(numel(ATPY_STARRYNITE_EVENT_TRACE.snapshots));
            ATPY_STARRYNITE_EVENT_TRACE.finished = true;
            output = [];

        case 'result'
            require_argument_count(action, varargin, 0);
            require_started_trace();
            if ~ATPY_STARRYNITE_EVENT_TRACE.finished
                error('ATPy:StarryNiteOracle:UnfinishedEventTrace', ...
                    'Event trace must be finalized before export.');
            end
            expected = size( ...
                ATPY_STARRYNITE_EVENT_TRACE.classification_table, 1) + 2;
            if ATPY_STARRYNITE_EVENT_TRACE.snapshot_count ~= expected
                error('ATPy:StarryNiteOracle:InvalidEventTraceShape', ...
                    ['Event trace needs initial, one pre-classifier snapshot ', ...
                     'per event, and final.']);
            end
            output = ATPY_STARRYNITE_EVENT_TRACE;

        otherwise
            error('ATPy:StarryNiteOracle:InvalidEventTraceAction', ...
                'Unknown event trace action: %s', action);
    end
end


function table = lineage_pointer_snapshot(esequence)
    if ~iscell(esequence) || isempty(esequence)
        error('ATPy:StarryNiteOracle:InvalidEventTraceState', ...
            'Lineage event state must be a non-empty cell array.');
    end
    counts = zeros(numel(esequence), 1);
    for time = 1:numel(esequence)
        detected = esequence{time};
        required = {'finalpoints', 'delete', 'pred', 'pred_time', 'suc', 'suc_time'};
        for index = 1:numel(required)
            if ~isfield(detected, required{index})
                error('ATPy:StarryNiteOracle:InvalidEventTraceState', ...
                    'esequence{%d}.%s is required for event tracing.', ...
                    time, required{index});
            end
        end
        counts(time) = size(detected.finalpoints, 1);
    end
    table = zeros(sum(counts), 9);
    output_row = 0;
    for time = 1:numel(esequence)
        detected = esequence{time};
        count = counts(time);
        if numel(detected.delete) ~= count || numel(detected.pred) ~= count || ...
                numel(detected.pred_time) ~= count || ...
                ~isequal(size(detected.suc), [count, 2]) || ...
                ~isequal(size(detected.suc_time), [count, 2])
            error('ATPy:StarryNiteOracle:InvalidEventTraceState', ...
                'Lineage pointer arrays do not match the detection count.');
        end
        for node = 1:count
            output_row = output_row + 1;
            predecessor = normalized_reference( ...
                detected.pred_time(node), detected.pred(node));
            successor1 = normalized_reference( ...
                detected.suc_time(node, 1), detected.suc(node, 1));
            successor2 = normalized_reference( ...
                detected.suc_time(node, 2), detected.suc(node, 2));
            table(output_row, :) = [ ...
                time - 1, node - 1, double(logical(detected.delete(node))), ...
                predecessor, successor1, successor2];
        end
    end
end


function reference = normalized_reference(time, node)
    time = double(time);
    node = double(node);
    if time <= 0 || node <= 0
        if time ~= -1 || node ~= -1
            error('ATPy:StarryNiteOracle:InvalidEventTraceReference', ...
                'Absent lineage references must use the -1/-1 sentinel.');
        end
        reference = [-1, -1];
        return;
    end
    if time ~= fix(time) || node ~= fix(node)
        error('ATPy:StarryNiteOracle:InvalidEventTraceReference', ...
            'Lineage references must be integer indices.');
    end
    reference = [time - 1, node - 1];
end


function require_started_trace()
    global ATPY_STARRYNITE_EVENT_TRACE; %#ok<GVMIS>
    if isempty(ATPY_STARRYNITE_EVENT_TRACE) || ...
            ~isstruct(ATPY_STARRYNITE_EVENT_TRACE) || ...
            ~isfield(ATPY_STARRYNITE_EVENT_TRACE, 'schema_version')
        error('ATPy:StarryNiteOracle:MissingEventTrace', ...
            'Event trace was used before initialization.');
    end
end


function require_argument_count(action, values, expected)
    if numel(values) ~= expected
        error('ATPy:StarryNiteOracle:InvalidEventTraceArguments', ...
            'Event trace action %s expects %d argument(s).', action, expected);
    end
end


function result = require_positive_integer(value, label)
    result = require_nonnegative_integer(value, label);
    if result < 1
        error('ATPy:StarryNiteOracle:InvalidEventTraceInteger', ...
            '%s must be positive.', label);
    end
end


function result = require_nonnegative_integer(value, label)
    if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value) || ...
            value ~= fix(value) || value < 0
        error('ATPy:StarryNiteOracle:InvalidEventTraceInteger', ...
            '%s must be a non-negative integer.', label);
    end
    result = double(value);
end


function result = require_class(value, label, allow_nan)
    if allow_nan && isnumeric(value) && isscalar(value) && isnan(value)
        result = NaN;
        return;
    end
    result = require_nonnegative_integer(value, label);
    if ~ismember(result, 0:3)
        error('ATPy:StarryNiteOracle:InvalidEventTraceClass', ...
            '%s must be 0, 1, 2, or 3.', label);
    end
end


function result = require_logical_scalar(value, label)
    if islogical(value) && isscalar(value)
        result = value;
        return;
    end
    if isnumeric(value) && isscalar(value) && isfinite(value) && ...
            ismember(value, [0, 1])
        result = logical(value);
        return;
    end
    error('ATPy:StarryNiteOracle:InvalidEventTraceLogical', ...
        '%s must be a logical scalar.', label);
end
