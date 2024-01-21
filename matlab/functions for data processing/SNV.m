function [data_snv] = SNV(data)
    mean_row = mean(data,2);
    std_spectra = std(data,[],2);
    % SNV preprocessing of spectral data for each sample
    data_snv = zeros(size(data));
    for i = 1:size(data, 1)
        data_snv(i,:) = (data(i,:) - mean_row(i)) ./ std_spectra(i);
    end
end