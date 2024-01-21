function Y = ELM_Predict(P, IW, B, LW, TF, TYPE)
%{
Input:
    P   - Input of Training Set(R*Q)
    IW  - Input (N*R)
    B   - Bias (N*1)
    LW  - Layer Weight (N*S)
    TF  - Transfer Function:
          'sig' for Sigmoidal function (default)
          'sin' for Sine function
          'hardlim' for Hardlim function
    TYPE - Regression (0, default) or Classification (1)
Output:
    Y   - Simulate Output (S*Q)
%}

% Calculate the Layer Output H
Q = size(P, 2);                 % Q = the number of columns of the matrix
BiasMatrix = repmat(B, 1, Q);   % repmat reorganise B to get the bias matrix B after arranging it by 1*Q
tempH = IW * P + BiasMatrix;    % Calculate output = input * weights + bias, subsequently put into activation function
% Selection of activation function (one of three)
switch TF
    case 'sig'                  % sigmoid
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'                  % sin
        H = sin(tempH);
    case 'hardlim'              % hardlim
        H = hardlim(tempH);
end

% Calculate the Simulate Output
Y = (H' * LW)';                 % output = hidden layer output * hidden layer node weights  
if TYPE == 1                    % Classification
    temp_Y = zeros(size(Y));
    for i = 1 : size(Y, 2)      % Number of columns of 1:Y
        [~, index] = max(Y(:, i));  % Get the index of the maximum value of each column of Y
        temp_Y(index, i) = 1;   % Distinguish categories by 0s and 1s
    end
    Y = vec2ind(temp_Y);        % Convert vector to index (get Y = index corresponding to maximum value of each column) (Y size: 1*size(Y))
end
end
