function run_starrynite_matlab_oracle_batch(manifest_path, starrynite_root)
%RUN_STARRYNITE_MATLAB_ORACLE_BATCH Run many requests in one MATLAB process.
%
% The manifest is a MAT-file with equally sized request_paths and result_paths
% cell arrays. Each request saves its own structured success or error result.
% Failures are caught so later trials still run and the Python caller can report
% every invalid point in a parameter sweep at once.

    loaded = load(manifest_path, 'request_paths', 'result_paths');
    if ~isfield(loaded, 'request_paths') || ~isfield(loaded, 'result_paths')
        error('ATPy:StarryNiteOracle:InvalidBatchManifest', ...
            'Batch manifest must contain request_paths and result_paths.');
    end
    request_paths = normalize_path_list(loaded.request_paths, 'request_paths');
    result_paths = normalize_path_list(loaded.result_paths, 'result_paths');
    if numel(request_paths) ~= numel(result_paths) || isempty(request_paths)
        error('ATPy:StarryNiteOracle:InvalidBatchManifest', ...
            'Batch path lists must have the same nonzero length.');
    end

    failure_count = 0;
    for index = 1:numel(request_paths)
        try
            run_starrynite_matlab_oracle( ...
                request_paths{index}, result_paths{index}, starrynite_root);
        catch exception
            failure_count = failure_count + 1;
            warning('ATPy:StarryNiteOracle:BatchTrialFailed', ...
                'Trial %d failed: %s', index, exception.message);
        end
    end
    if failure_count > 0
        warning('ATPy:StarryNiteOracle:BatchCompletedWithFailures', ...
            '%d of %d oracle trials failed.', failure_count, numel(request_paths));
    end
    close_parallel_pool();
end


function values = normalize_path_list(value, label)
    if isstring(value)
        values = cellstr(value(:));
    elseif ischar(value)
        values = cellstr(value);
    elseif iscell(value)
        values = value(:);
    else
        error('ATPy:StarryNiteOracle:InvalidBatchManifest', ...
            '%s must be a string, character array, or cell array.', label);
    end
    for index = 1:numel(values)
        item = values{index};
        if isstring(item) && isscalar(item)
            item = char(item);
        end
        if ~ischar(item) || isempty(strtrim(item))
            error('ATPy:StarryNiteOracle:InvalidBatchManifest', ...
                '%s contains an invalid path at index %d.', label, index);
        end
        values{index} = item;
    end
end


function close_parallel_pool()
    % Explicit shutdown releases Windows handles to request MAT-files before
    % the Python temporary-directory context attempts cleanup.
    if license('test', 'Distrib_Computing_Toolbox')
        pool = gcp('nocreate');
        if ~isempty(pool)
            delete(pool);
        end
    end
end
