function [data_IF, labels_IF] = IForest(data, labels, ratio)
    % 拟合孤立森林
    [forest, isAnomaly, scores] = iforest(data, 'NumLearners', 100, 'ContaminationFraction', ratio);
    % 预测标签(正常-0，异常-1)
    labels_IF = isAnomaly;
    % 获取正常样本
    data_IF = data(labels_IF == 0, :);
    labels_IF = labels(labels_IF == 0);
end
