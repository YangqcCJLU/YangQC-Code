function [datacube, wavelengths, img_binary, mask, row, col, bands, num] = HSI_load(path, band_Otsu)
    % Input:
    %       path-高光谱文件路径
    %       band_Otsu-阈值分割所用波段
    % Output:
    %       datacube-数据立方体
    %       wavelengths-波段信息
    %       img_binary-二值化掩膜
    %       mask-掩膜内所有样本像素点的光谱数据(二维光谱矩阵,行-样本;列-波段)

    hcube = hypercube(path);
    datacube = hcube.DataCube;
    wavelengths = hcube.Wavelength;
    img = datacube(:, :, band_Otsu);
    min_value = min(img(:));
    max_value = max(img(:));
    img_scaled = uint8((double(img) - double(min_value)) * 255.0 / double(max_value - min_value));
    % 利用Otsu阈值法对图像进行二值化处理
    level = graythresh(img_scaled);
    img_binary = imbinarize(img_scaled, level);

    %% 图像分割可视化
    figure;
    subplot(1,2,1);
    imshow(img_binary);
    subplot(1,2,2);
    imshow(labeloverlay(img_scaled,img_binary));

    %% 相乘取掩膜
    [row,col,bands] = size(datacube);
    sampled_data_3d = zeros(size(datacube));
    for b = 1:bands
        sampled_data_3d(:,:,b) = img_binary .* datacube(:,:,b);
    end
    sampled_data_2d = reshape(sampled_data_3d,[row*col,bands]);
    for i = 1:size(sampled_data_2d,1)
        sampled_data_2d(i, 151+49) = sampled_data_2d(i, 151+49) - 0.006;
    end


    num = find(img_binary == 1);
    mask = sampled_data_2d(num,:);
end