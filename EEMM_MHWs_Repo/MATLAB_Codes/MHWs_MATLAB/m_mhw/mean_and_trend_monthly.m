function [mean_metric, monthly_metric, trend_metric, p_metric] = mean_and_trend_monthly(MHW, mhw_ts, data_start, varargin)
% mean_and_trend - calculating mean states and trends of event metrics
% Syntax
% [mean_metric, monthly_metric, trend_metric, p_metric] = mean_and_trend(MHW, mhw_ts, data_start);
% [mean_metric, monthly_metric, trend_metric, p_metric] = mean_and_trend(MHW, mhw_ts, data_start, 'Metric', 'Duration');
%
% Description
% [mean_metric, monthly_metric, trend_metric, p_metric] = mean_and_trend(MHW, mhw_ts, data_start)
% returns the mean states and monthly trends of MHW/MCS frequency (number
% of events per month) based on event table MHW and MHW/MCS time series
% MHW_TS. The start year (DATA_START) is the first year of MHW_TS.
%
% [mean_metric, monthly_metric, trend_metric, p_metric] = mean_and_trend(MHW, mhw_ts, data_start, 'Metric', 'Duration')
% returns the mean states and monthly trends of MHW/MCS duration. 
%
% Input Arguments
%
% MHW, mhw_ts - Outputs from function detect
%
% data_start - A numeric value indicating the start year of MHW_TS.
%
% 'Metric' - Default is 'Frequency'. The metric for which mean states and annual trends are
% calculated.
% 'Frequency' - The monthly number of events.
% 'Duration' - The duration of events.
% 'MaxInt' - The maximum intensity of events.
% 'MeanInt' - The mean intensity of events.
% 'CumInt' - The cumulative intensity of events.
% 'Days' - The monthly total MHW/MCS days.
%
% Output Arguments
% 
% mean_metric - A 2D numeric matrix (m-by-n) containing the mean states
% of MHW/MCS metrics.
% 
% monthly_metric - A 3D numeric matrix (m-by-n-by-M) containing monthly
% mean MHW/MCS metrics, where M indicates the number of months based on the
% start year DATA_START and the size of MHW_TS.
%
% trend_metric - A 2D numeric matrix (m-by-n) containing linear trend
% calculated from MONTHLY_METRIC in unit of 'unit of metric/month'.
%
% p_metric - A 2D numeric matrix (m-by-n) containing p value of
% TREND_METRIC.

paramNames = {'Metric'};
defaults   = {'Frequency'};

[vMetric] = internal.stats.parseArgs(paramNames, defaults, varargin{:});

MetricNames = {'Frequency', 'Duration', 'MaxInt', 'MeanInt', 'CumInt', 'Days'};
vMetric = internal.stats.getParamVal(vMetric, MetricNames, '''Metric''');

[x, y, ~] = size(mhw_ts);

switch vMetric
    case 'Duration'
        % Similar modifications for other metrics
        % Duration specific code goes here
    case 'MaxInt'
        % MaxInt specific code goes here
    case 'MeanInt'
        % MeanInt specific code goes here
    case 'CumInt'
        % CumInt specific code goes here
    case 'Days'
        MHW = MHW{:,:};
        period_used = datenum(data_start, 1, 1):(datenum(data_start, 1, 1) + size(mhw_ts, 3) - 1);
        period_used = datevec(period_used);
        
        % Create an array of unique months and years
        [years_months, ~, unique_months_indices] = unique(period_used(:, 1:2), 'rows');
        
        mean_metric = NaN(x, y);
        trend_metric = NaN(x, y);
        p_metric = NaN(x, y);
        monthly_metric = NaN(x, y, length(unique_months_indices));
        
        loc_full = unique(MHW(:, 8:9), 'rows');
        
        for m = 1:size(loc_full, 1)
            loc_here = loc_full(m, :);
            
            for i = 1:length(unique_months_indices)
                year_month_here = years_months(i, :);
                mhw_here = squeeze(mhw_ts(loc_here(1), loc_here(2), ...
                    (datenum(year_month_here(1), year_month_here(2), 1): ...
                    datenum(year_month_here(1), year_month_here(2), eomday(year_month_here(1), year_month_here(2)))) - datenum(data_start, 1, 1) + 1));
                
                monthly_metric(loc_here(1), loc_here(2), i) = nansum(mhw_here ~= 0 & ~isnan(mhw_here));
            end
            
            ts_here = squeeze(monthly_metric(loc_here(1), loc_here(2), :));
            mdl = fitlm((1:length(ts_here))', ts_here);
            trend_metric(loc_here(1), loc_here(2)) = mdl.Coefficients.Estimate(2);
            p_metric(loc_here(1), loc_here(2)) = mdl.Coefficients.pValue(2);
        end
    case 'Frequency'
        MHW = MHW{:,:};
        full_mhw_start = datevec(num2str(MHW(:, 1)), 'yyyymmdd');
        full_mhw_end = datevec(num2str(MHW(:, 2)), 'yyyymmdd');
        
        period_used = datenum(data_start, 1, 1):(datenum(data_start, 1, 1) + size(mhw_ts, 3) - 1);
        period_used = datevec(period_used);
        
        % Create an array of unique months and years
        [years_months, ~, unique_months_indices] = unique(period_used(:, 1:2), 'rows');
        
        mean_metric = NaN(x, y);
        trend_metric = NaN(x, y);
        p_metric = NaN(x, y);
        monthly_metric = NaN(x, y, length(unique_months_indices));
        
        loc_full = unique(MHW(:, 8:9), 'rows');
        
        for m = 1:size(loc_full, 1)
            loc_here = loc_full(m, :);
            MHW_here = MHW(MHW(:, 8) == loc_here(1) & MHW(:, 9) == loc_here(2), :);
            
            mean_metric(loc_here(1), loc_here(2)) = (size(MHW_here, 1) / length(years_months));
            
            for i = 1:length(unique_months_indices)
                year_month_here = years_months(i, :);
                judge_1 = (datenum(num2str(MHW_here(:, 1)), 'yyyymmdd') >= datenum(year_month_here(1), year_month_here(2), 1)) & ...
                          (datenum(num2str(MHW_here(:, 2)), 'yyyymmdd') <= datenum(year_month_here(1), year_month_here(2), eomday(year_month_here(1), year_month_here(2))));
                MHW_judge = MHW_here(judge_1, :);
                
                monthly_metric(loc_here(1), loc_here(2), i) = size(MHW_judge, 1);
            end
            
            ts_here = squeeze(monthly_metric(loc_here(1), loc_here(2), :));
            mdl = fitlm((1:length(ts_here))', ts_here);
            trend_metric(loc_here(1), loc_here(2)) = mdl.Coefficients.Estimate(2);
            p_metric(loc_here(1), loc_here(2)) = mdl.Coefficients.pValue(2);
        end
end
