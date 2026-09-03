function predicted_class = atpy_trace_predictBifurcationTypeSinglemodel(varargin)
%ATPY_TRACE_PREDICTBIFURCATIONTYPESINGLEMODEL Trace one upstream prediction.

    global computedclassificationvector; %#ok<GVMIS>
    global refclassificationvector; %#ok<GVMIS>

    esequence = evalin('caller', 'esequence');
    time = evalin('caller', 't');
    if evalin('caller', 'exist(''mind'', ''var'')') == 1
        parent = evalin('caller', 'mind');
        classifier_round = 2;
    else
        parent = evalin('caller', 'i');
        classifier_round = 1;
    end
    before_computed = numel(computedclassificationvector);
    before_effective = numel(refclassificationvector);
    predicted_class = predictBifurcationTypeSinglemodel(varargin{:});
    if numel(computedclassificationvector) ~= before_computed + 1 || ...
            numel(refclassificationvector) ~= before_effective + 1
        error('ATPy:StarryNiteOracle:UnexpectedClassifierTrace', ...
            'Single-model classifier did not append exactly one class pair.');
    end
    atpy_starrynite_trace( ...
        'classification', esequence, time, parent, classifier_round, ...
        predicted_class, computedclassificationvector(end), ...
        refclassificationvector(end), varargin{end}, 0);
end
