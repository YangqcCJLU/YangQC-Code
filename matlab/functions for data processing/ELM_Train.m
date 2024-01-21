function [IW, B, LW, TF, TYPE] = ELM_Train(P, T, N, TF, TYPE)
%{
Input:
    P   - Input of Training Set(Each column represents a sample)(R*Q)
    T   - Output of Training Set(S*Q)
    N   - Number of Hidden Neurons
    TF  - Transfer Function:
          'sig' for Sigmoidal function (default)
          'sin' for Sine function
          'hardlim' for Hardlim function
    TYPE - Regression (0, default) or Classification (1)
Output:
    IW  - Input Weight (N*R)
    B   - Bias (N*1)
    LW  - Layer Weight (N*S)
%}
[R, Q] = size(P);       % Get the ranks R, Q of the input matrix P of the training set
if TYPE  == 1           % Classification
    T  = ind2vec(T);    % Convert indexes to vectors
end

% Step1(randomly generated w, β)
% Randomly Generate the Input Weight
IW = rand(N, R) * 2 - 1;
% Randomly Generate the Bias
B = rand(N, 1);
% repmat reorganise B to get the bias matrix B after arranging it by 1*Q
BiasMatrix = repmat(B, 1, Q);

% Step2(Calculate the output H of the implicit layer)
% Calculate the Layer Output H
tempH = IW * P + BiasMatrix;
% Selection of activation function (one of three)
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end

% Step3(Solve for the weights β between the implicit and output layers, 
% since the matrix paradigm of Hβ-T is 0)

% Calculate the Output Weight
LW = pinv(H') * T'; % pinv: return the Moore-Penrose pseudo-inverse of the matrix
end
