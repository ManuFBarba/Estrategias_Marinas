%%% MATLAB Script para el cálculo de métricas de MHWs mensuales en las
%%% demarcaciones marinas españolas (EEMM)
clear;
close all;
clc;

year_ini = 1993;

%% Mascara ya reducida por el STRIDE (punto inicial, tamaño matriz final, salto en la lectura)
sst_mask = ncread('...\bottomT_GLORYS_NA_clipped.nc', 'bottomT', [1,1,1], [626,301,30], [1,1,1]);
% sst_maskfull=sst_mask-273.15; clear sst_mask
sst_maskfull = sst_mask; clear sst_mask
mascara = nanmean(sst_maskfull, 3); 
mascara(isfinite(mascara) == 1) = 1; clear sst_maskfull

%% Creo las matrices para rellenar
months = 30 * 12; % para 30 años y 12 meses al año

MHW_cnt_ts = NaN(size(mascara, 1), size(mascara, 2), months);
MHW_mean_ts = NaN(size(mascara, 1), size(mascara, 2), months);
MHW_max_ts = NaN(size(mascara, 1), size(mascara, 2), months);
MHW_cum_ts = NaN(size(mascara, 1), size(mascara, 2), months);
MHW_td_ts = NaN(size(mascara, 1), size(mascara, 2), months);
MHW_dur_ts = NaN(size(mascara, 1), size(mascara, 2), months);

%% Para crear la matriz cada 10 pixeles y todos los días
% sst_completa = NaN*ones(720,100,14610);
% for i=1:14610
%     i
%     sst_completa(:,:,i) = ncread('...\SST_Canary_clipped.nc', 'analysed_sst', [1,1,i], [240,181,1], [10,10,1]);
% end
sst_completa = ncread('...\bottomT_GLORYS_NA_clipped.nc', 'bottomT');
save('sst_completa.mat', 'sst_completa', '-v7.3'); % save data to a MAT-file in version 7.3 or later

load sst_completa.mat

%% Datos de tiempo, lat y lon 
time = ncread('...\bottomT_GLORYS_NA_clipped.nc', 'time');
% fecha = datenum(1982,1,1) + (double(time)); clear time

lat = ncread('...\bottomT_GLORYS_NA_clipped.nc', 'latitude');
latitude = lat(1:1:301); clear lat

lon = ncread('...\bottomT_GLORYS_NA_clipped.nc', 'longitude');
longitude = lon(1:1:626); clear lon

%% Eliminación de la tendencia lineal pixel a pixel
for i = 2:1:625
    for j = 2:1:300
        disp([i, j])
        
        sst = sst_completa(i-1:i+1,j-1:j+1,:);
        sst_full = sst; clear sst
        
        if isfinite(mascara(i,j)) == 1
            % Ajuste de una tendencia lineal a los datos de temperatura
            Tt = squeeze(sst_full(2,2,:)); % Serie temporal en el pixel central
            t = 1:length(Tt);
            p = polyfit(t, Tt, 1); % Ajuste lineal
            Xt = polyval(p, t); % Tendencia lineal
            Tt_detrended = Tt - Xt; % Temperatura sin tendencia
            
            sst_full(2,2,:) = Tt_detrended; % Reemplazar la serie original con la serie sin tendencia
            
            %Cambiar climatología, según se requiera
            [MHW, mclim, m90, mhw_ts] = detect(sst_full, datenum(1993,1,1):datenum(2022,12,31), datenum(1993,1,1), datenum(2022,12,31), datenum(1993,1,1), datenum(2022,12,31)); % Toma alrededor de 30 segundos.
            
            % Variables de estado y tendencias para siete métricas diferentes
            [MHW_cnt_ts_i] = mean_and_trend_monthly(MHW, mhw_ts, year_ini, 'Metric', 'Frequency');
            [MHW_mean_ts_i] = mean_and_trend_monthly(MHW, mhw_ts, year_ini, 'Metric', 'MeanInt');
            [MHW_max_ts_i] = mean_and_trend_monthly(MHW, mhw_ts, year_ini, 'Metric', 'MaxInt');
            [MHW_cum_ts_i] = mean_and_trend_monthly(MHW, mhw_ts, year_ini, 'Metric', 'CumInt');
            [MHW_td_ts_i] = mean_and_trend_monthly(MHW, mhw_ts, year_ini, 'Metric', 'Days');
            [MHW_dur_ts_i] = mean_and_trend_monthly(MHW, mhw_ts, year_ini, 'Metric', 'Duration');
          
            MHW_cnt_ts(i,j,:) = squeeze(MHW_cnt_ts_i(2,2,:));
            MHW_mean_ts(i,j,:) = squeeze(MHW_mean_ts_i(2,2,:));
            MHW_max_ts(i,j,:) = squeeze(MHW_max_ts_i(2,2,:));
            MHW_cum_ts(i,j,:) = squeeze(MHW_cum_ts_i(2,2,:));
            MHW_td_ts(i,j,:) = squeeze(MHW_td_ts_i(2,2,:));
            MHW_dur_ts(i,j,:) = squeeze(MHW_dur_ts_i(2,2,:));

            clear MHW_cnt_ts_i MHW_mean_ts_i MHW_max_ts_i MHW_cum_ts_i MHW_td_ts_i MHW_dur_ts_i                 
            clear sst_full
        end
    end
    save total_bottom_NA_MODEL_monthly
end

%% Función modificada mean_and_trend_monthly
function [metric_ts] = mean_and_trend_monthly(MHW, mhw_ts, year_ini, varargin)
    % Adaptar la función para calcular métricas mensuales

    % Parámetros
    args = inputParser;
    addParameter(args, 'Metric', 'Frequency');
    parse(args, varargin{:});
    metric = args.Results.Metric;

    % Inicialización de variables
    months = 12 * 30; % para 30 años y 12 meses al año
    metric_ts = NaN(size(mhw_ts,1), size(mhw_ts,2), months);

    % Cálculo de la métrica mensual
    for i = 1:size(mhw_ts,1)
        for j = 1:size(mhw_ts,2)
            for month = 1:months
                % Filtrar los datos correspondientes al mes
                current_month_data = mhw_ts(i, j, (month-1)*30+1 : month*30);
                
                % Calcular la métrica según el tipo
                switch metric
                    case 'Frequency'
                        metric_ts(i, j, month) = sum(current_month_data > 0);
                    case 'MeanInt'
                        metric_ts(i, j, month) = mean(current_month_data(current_month_data > 0));
                    case 'MaxInt'
                        metric_ts(i, j, month) = max(current_month_data);
                    case 'CumInt'
                        metric_ts(i, j, month) = sum(current_month_data);
                    case 'Days'
                        metric_ts(i, j, month) = sum(current_month_data > 0);
                    case 'Duration'
                        metric_ts(i, j, month) = mean(diff(find([0; current_month_data > 0; 0]) == 1) - 1);
                end
            end
        end
    end
end
