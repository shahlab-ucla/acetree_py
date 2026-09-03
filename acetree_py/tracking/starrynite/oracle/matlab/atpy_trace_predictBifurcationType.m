function predicted_class = atpy_trace_predictBifurcationType(varargin)
%ATPY_TRACE_PREDICTBIFURCATIONTYPE Trace one historical multi-model call.

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
    predicted_class = predictBifurcationType(varargin{:});
    if numel(computedclassificationvector) ~= before_computed + 1 || ...
            numel(refclassificationvector) ~= before_effective + 1
        error('ATPy:StarryNiteOracle:UnexpectedClassifierTrace', ...
            'Multi-model classifier did not append exactly one class pair.');
    end
    atpy_starrynite_trace( ...
        'classification', esequence, time, parent, classifier_round, ...
        predicted_class, computedclassificationvector(end), ...
        refclassificationvector(end), varargin{end}, 1);
end
