clc;
clear;
close all;

%% ---------------------------Load HSI Data---------------------------
[datacube1, wavelengths, img_binary1, mask1, ~, ~, ~, ~] = HSI_load("...\class1.hdr", 106);
[datacube2, ~, img_binary2, mask2, ~, ~, ~, ~] = HSI_load("...\class2.hdr", 106);
[datacube3, ~, img_binary3, mask3, ~, ~, ~, ~] = HSI_load("...\class3.hdr", 106);
[datacube4, ~, img_binary4, mask4, ~, ~, ~, ~] = HSI_load("...\class4.hdr", 106);
[datacube5, ~, img_binary5, mask5, ~, ~, ~, ~] = HSI_load("...\class5.hdr", 106);

mask1 = mask1(:,50:250);mask2 = mask2(:,50:250);
mask3 = mask3(:,50:250);mask4 = mask4(:,50:250);mask5 = mask5(:,50:250);
mask = [mask1;mask2;mask3;mask4;mask5];

wavelength = wavelengths(50:250);

labels1 = ones(length(mask1),1);labels2 = 2*ones(length(mask2),1);
labels3 = 3*ones(length(mask3),1);labels4 = 4*ones(length(mask4),1);
labels5 = 5*ones(length(mask5),1);
labels = [labels1;labels2;labels3;labels4;labels5];

%% ----------------------------Isolation Forest----------------------------
data_dIF = mask;
[data_Rout1, labels1] = IForest(data_dIF(1:length(mask1),:), labels1, 0.1);
[data_Rout2, labels2] = IForest(data_dIF(length(mask1)+1:length(mask1)+length(mask2),:), labels2, 0.1);
[data_Rout3, labels3] = IForest(data_dIF(length(mask1)+length(mask2)+1:length(mask1)+length(mask2)+length(mask3),:), labels3, 0.1);
[data_Rout4, labels4] = IForest(data_dIF(length(mask1)+length(mask2)+length(mask3)+1:length(mask1)+length(mask2)+length(mask3)+length(mask4),:), labels4, 0.1);
[data_Rout5, labels5] = IForest(data_dIF(length(mask1)+length(mask2)+length(mask3)+length(mask4)+1:length(mask1)+length(mask2)+length(mask3)+length(mask4)+length(mask5),:), labels5, 0.1);
data_Rout = [data_Rout1;data_Rout2;data_Rout3;data_Rout4;data_Rout5];
labels = [labels1;labels2;labels3;labels4;labels5];

%% -----------------------------preprocessing-----------------------------
% SG
data_SG = SG_smooth(data_Rout, 3, 9);
% SNV
data_snv = SNV(data_SG);

% Spectral Comparison - Random points
idx = randperm(length(data_Rout1));
spec = data_Rout;
spec1 = spec(1:length(data_Rout1),:);
spec2 = spec(1+length(data_Rout1):length(data_Rout1)+length(data_Rout2),:);
spec3 = spec(1+length(data_Rout1)+length(data_Rout2):length(data_Rout1)+length(data_Rout2)+length(data_Rout3),:);
spec4 = spec(1+length(data_Rout1)+length(data_Rout2)+length(data_Rout3):length(data_Rout1)+length(data_Rout2)+length(data_Rout3)+length(data_Rout4),:);
spec5 = spec(1+length(data_Rout1)+length(data_Rout2)+length(data_Rout3)+length(data_Rout4):length(data_Rout1)+length(data_Rout2)+length(data_Rout3)+length(data_Rout4)+length(data_Rout5),:);
num = 100;
raw = [spec1(idx(1:num),:);spec2(idx(1:num),:);spec3(idx(1:num),:);spec4(idx(1:num),:);spec5(idx(1:num),:)];
plot(wavelength, raw);
xlabel('Wavelength(nm)');
ylabel('Reflectance');
xlim([480, 927]);
% Spectral Comparison - mean spectra
mean1 = mean(spec1);mean2 = mean(spec2);mean3 = mean(spec3);mean4 = mean(spec4);mean5 = mean(spec5);
std1 = std(spec1);std2 = std(spec2);std3 = std(spec3);std4 = std(spec4);std5 = std(spec5);
plot(wavelength, mean1, 'g');hold on;
plot(wavelength, mean2, 'b');hold on;
plot(wavelength, mean3, 'c');hold on;
plot(wavelength, mean4, 'y');hold on;
plot(wavelength, mean5, 'r');
xlabel('Wavelength(nm)');
ylabel('Reflectance');
legend('none','procymidone','oxytetracycline','indoleacetic acid', 'gibberellin');
xlim([480, 927]);

%% ----------------------------feature extraction----------------------------
% PCA
[data_PCA, proj_pca] = PCA_DR(data_snv, labels, 98);
% SPA
[data_SPA, bands_spa, ~] = SPA_DR(data_snv,labels,0.8,20,100,0,wavelength);
% iRF
data_RF = iRF(data_snv,labels,100,5,5,16,'center');
plot(1:100, data_RF.RMSEP, LineWidth=1.2);hold on;
[~, idx_min] = min(data_RF.RMSEP);
plot(idx_min, data_RF.RMSEP(idx_min),'s',LineWidth=1.6);
xlabel('The union of the ranked intervals');
ylabel('RMSECV');

plot(1:length(data_RF.probability), data_RF.probability, LineWidth=1.2);
xlabel('The union of the ranked intervals');
ylabel('Selection probability');

% Initialize an empty vector to store all elements
allElements = [];
% Loop over each cell and add its elements to allElements.
for i = 1:numel(intervals)
    currentVector = intervals{i};
    allElements = [allElements, currentVector];
end
% Use the unique function to get a vector with no duplicate elements
bands_iRF = unique(allElements);
data_iRF = data_stand(:,bands_iRF);
x = wavelength(bands_iRF);
y = data_SG(1, bands_iRF);
plot(wavelength, data_SG(1,:), "LineWidth", 1.2, "Color","blue");
hold on;
for i = 1:length(x)
    plot([x(i),x(i)], [0, y(i)], 'r-', 'LineWidth', 1);
end
xlabel('Wavelength(nm)');
ylabel('Reflectance');
legend('Spectrum','Selected bands', Location='northwest')
xlim([480, 927]);

%% -----------------------classification model-------------------------
% ----------------------------load ROIs----------------------------
ROI11 = readtable("...\water_t1.xlsx");
ROI12 = readtable("...\water_t2.xlsx");
ROI21 = readtable("...\fumeili_t1.xlsx");
ROI22 = readtable("...\fumeili_t2.xlsx");
ROI31 = readtable("...\tumeisu_t1.xlsx");
ROI32 = readtable("...\tumeisu_t2.xlsx");
ROI41 = readtable("...\IAA_t1.xlsx");
ROI42 = readtable("...\IAA_t2.xlsx");
ROI51 = readtable("...\chimeisu_t1.xlsx");
ROI52 = readtable("...\chimeisu_t2.xlsx");

ROI11 = table2array(ROI11(2:end, 3:end));
ROI12 = table2array(ROI12(2:end, 3:end));
ROI21 = table2array(ROI21(2:end, 3:end));
ROI22 = table2array(ROI22(2:end, 3:end));
ROI31 = table2array(ROI31(2:end, 3:end));
ROI32 = table2array(ROI32(2:end, 3:end));
ROI41 = table2array(ROI41(2:end, 3:end));
ROI42 = table2array(ROI42(2:end, 3:end));
ROI51 = table2array(ROI51(2:end, 3:end));
ROI52 = table2array(ROI52(2:end, 3:end));

ROI1 = [ROI11;ROI12];ROI2 = [ROI21;ROI22];
ROI3 = [ROI31;ROI32];ROI4 = [ROI41;ROI42];
ROI5 = [ROI51;ROI52];
ROI = [ROI1;ROI2;ROI3;ROI4;ROI5];
ROI = ROI(:,50:250);
% SG
data_SG_t = SG_smooth(ROI, 3, 9);
% zscore
data_snv_t = SNV(data_SG_t);

%% ----------------------------feature extraction----------------------------
% PCA
data_PCA_t = data_snv_t * proj_pca;
% iRF
data_iRF_t = data_snv_t(:, bands_iRF);
% SPA
data_SPA_t = data_snv_t(:,bands_spa);

labels1_t = ones(length(ROI1),1);labels2_t = 2*ones(length(ROI2),1);
labels3_t = 3*ones(length(ROI3),1);labels4_t = 4*ones(length(ROI4),1);
labels5_t = 5*ones(length(ROI5),1);
labels_t = [labels1_t;labels2_t;labels3_t;labels4_t;labels5_t];

%% SVM
cmd = ['-t 2','-h 0',' -c 72.831',' -g 76.568'];
% train
model = svmtrain(labels, data_PCA, cmd);
[T_sim_1, accuracy1,~] = svmpredict(labels,data_snv,model);
[T_sim_2, accuracy2,~] = svmpredict(labels_t,data_snv_t,model);

%% ELM
P_train = data_snv';
T_train = labels';
P_test = data_snv_t';
T_test = labels_t';
% train(100 number of hidden neurons, sigmoid function, Classification)
[IW, B, LW, TF, TYPE] = ELM_Train(P_train, T_train, 2000, 'sig', 1);
% test
T_sim_1 = ELM_Predict(P_train, IW, B, LW, TF, TYPE);  % results
T_sim_2 = ELM_Predict(P_test, IW, B, LW, TF, TYPE);
T_sim_2 = T_sim_2';

% The impact of the number of neurons on the accuracy of ELM
R = zeros(100, 2);
for i = 1 : 100
    % train(number of neurons:50~5000,step=50)
    [IW, B, LW, TF, TYPE] = ELM_Train(P_train, T_train, 50*i, 'sig', 1);
    % test
    T_sim_1 = ELM_Predict(P_train, IW, B, LW, TF, TYPE);
    T_sim_2 = ELM_Predict(P_test, IW, B, LW, TF, TYPE);
    % acc(train)
    k1 = length(find(T_train == T_sim_1));
    n1 = length(T_train);
    Accuracy_1 = k1 / n1 * 100;
    % acc(test)
    k2 = length(find(T_test == T_sim_2));
    n2 = length(T_test);
    Accuracy_2 = k2 / n2 * 100;
    R(i, :) = [Accuracy_1 Accuracy_2];
end
plot(50 * (1 : 100), R(:, 1), 'b:o', 'LineWidth', 0.6);hold on;
plot(50 * (1 : 100), R(:, 2), 'r:o', 'LineWidth', 0.6);
xlabel('Number of neurons');
ylabel('Accuracy/%');
legend('train','test');

%% confusion matrix
cls = unique(labels_t);
num_cls = length(cls);            % number of classes
num_ref = zeros(num_cls,1);
cf_mt = zeros(num_cls);           % initialization

for i = 1:num_cls
    ind = find(T_sim_2 == cls(i));  % Position of test data type i(predicted label=label)
    num_ref(i) = length(ind);       % Number of test data type i           
    for j = 1:num_cls
        ind1 = find(labels_t == cls(i) & T_sim_2 == cls(j));
        cf_mt(i,j) = length(ind1);
    end
end

PA = zeros(num_cls,1);              % Initialization of evaluation indicators(PA%UA)
UA = zeros(1,num_cls);
for i = 1:num_cls
    PA(i) = cf_mt(i,i)/sum(cf_mt(i,:));
    UA(i) = cf_mt(i,i)/sum(cf_mt(:,i));
end

cf_mt(isnan(cf_mt)) = 0;            % Determines which array elements are NaN, and assigns 0 if any.

RT_A = sum(cf_mt,2);                % Summing up the rows            
CT_A = sum(cf_mt,1);                % Summing up the columns            
N_A = size(labels_t,1);             % number of samples
% Complete the confusion matrix
cf_mt(num_cls + 1,:) = CT_A;
cf_mt(num_cls + 2,1:num_cls) = UA;
cf_mt(1:num_cls,num_cls + 1) = RT_A';
cf_mt(1:num_cls,num_cls + 2) = PA;
cf_mt(num_cls + 1,num_cls + 1) = N_A;

% Kappa
PO_A = 0;
for index = 1:num_cls
    PO_A = PO_A + cf_mt(index,index)/N_A;  % overall classification accuracy, θ1
end
PC_A = 1/N_A^2 * sum(RT_A.*CT_A');         % Incidental consistency error, θ2
Kappa = (PO_A-PC_A)/(1-PC_A);

% plot confusion matrix
labels = categorical(labels_t);
T_sim_2 = categorical(T_sim_2);
figure;
plotconfusion(labels, T_sim_2);

%% ---------------Visualization of classification results----------------
[datacube, wavelengths, img_binary, mask_test, row, col, bands, num] = HSI_load("E:\11.12\test-new\test_all2.hdr", 106);
mask_test = mask_test(:,50:250);
mask_snv = SNV(SG_smooth(mask_test, 3, 9));

% *The two machine learning models are trained as above to obtain,
% *To improve efficiency, you can save the training results before loading them.
%% SVM
% load model(iRF-SVM).mat
virtual_labels = ones(length(mask_stand),1);
[predict_labels2, ~, ~]= svmpredict(virtual_labels, mask_snv, model);

%% ELM
% load model(IW-iRF-ELM).mat
% load model(B-iRF-ELM).mat
% load model(LW-iRF-ELM).mat

predict_labels2 = ELM_Predict(mask_snv', IW, B, LW, 'sig', 1);
predict_labels2 = predict_labels2';

% Here 'predict_labels2' can also be replaced with the output of the model in python, which can be visualized as well
figure;
multi_class = zeros(size(reshape(img_binary,[row*col,1])));
for i = 1:length(predict_labels2)
    multi_class(num(i)) = predict_labels2(i);
end
clusteredImage = zeros(size(img_binary));
for i = 0:5
    clusteredImage(multi_class == i) = i;
end
clusteredImage = clusteredImage + ones(size(clusteredImage));
% color distribution
customColors = [0,0,0;0,1,0;0,0,1;1,1,0;1,0,1;1,0,0];
% Convert marker images to RGB images
rgbImage = ind2rgb(clusteredImage, customColors);
h = imshow(rgbImage);
