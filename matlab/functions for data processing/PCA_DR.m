function [data_PCA, project_direction] = PCA_DR(data, labels, ratio)
    % input：data-Spectral Data Matrix
    %        labels-Category tags for visualization
    %        ratio-Cumulative variance as a percentage (%)
    % output：Result after dimensionality reduction

    [coeff, score, latent, ~, explained] = pca(data);
    % The number of principal components to be retained was selected based 
    % on the principal component variance contributions
    cumulative_explained = cumsum(explained);
    num_components = find(cumulative_explained >= ratio, 1);
    % Project the data onto the first num_components principal components
    project_direction = coeff(:, 1:num_components);
    data_PCA = data * project_direction;
    % visualization
    figure;
    pc1 = score(:,1);
    pc2 = score(:,2);
    pc3 = score(:,3);
    scatter3(pc1,pc2,pc3,[],labels,'o');
    xlabel('PC1');
    ylabel('PC2');
    zlabel('PC3');
    grid off;
end
