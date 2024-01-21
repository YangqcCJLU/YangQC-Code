function [data_SG] = SG_smooth(data, order, window_size)
    data_SG = zeros(size(data));
    for i = 1:size(data,1)
        % choose: Order, window size
        data_SG(i,:) = sgolayfilt(data(i,:),order,window_size);
    end
end