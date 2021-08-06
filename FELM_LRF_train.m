function [result, param] = FELM_LRF_train(X, T, param, C)
%FELM_LRF_TRAIN Summary of this function goes here
%   Detailed explanation goes here

[nX, hX, wX, cX] = size(X);

[nTrain, nClass] = size(T);

X = reshape(X, nTrain, hX*wX*cX);

image_size = hX;
input_ch = cX;

% Begins to generate random filters
fprintf('Begins to generate random filters...\n');

tic;
layer_param=param.network_params{1};
layer_param.input_ch = input_ch;
layer_param.image_size = image_size;

% randomly generate the weight matrix W
[W0, rf_index, pool_index, h_dim, tied_units] = gen_weights_my2(layer_param);

% update parameters for the next layer
image_size = h_dim;
input_ch = size(rf_index, 1)/(image_size^2);

% forwardprop X through current layer to generate input for the next
% layer
W = expand_rf(layer_param, h_dim, tied_units, W0);
W = full_size(W, rf_index);
[dummy, H] = two_layer_forwardprop(X, W, pool_index, layer_param.l1_act, layer_param.l2_act); 
time_network = toc;

% the training of the last layer (ELM-random) begins:
fprintf('The training of the last layer (ELM-random) begins:\n');

% calculating the output weight \beta and predict_Y-T
N = nTrain;
tic;

if param.isUseClassDistFuzzy > 0
    H = bsxfun(@times, param.u, H);
    T = bsxfun(@times, param.u, T);
end

% Compute Beta: output weight
if size(H,1) <= size(H,2)  % H is [N, K*(d-r+1)^2]
    BETA = H' * ((eye(N, N)/C + H*H') \ T); % A*inv(B)*C  --> A*(B\C)
else
    BETA = (eye(size(H,2))/C +H'*H) \ (H'*T); % inv(A)*B  --> A\B
end


if param.isUseTrainErrorFuzzy > 0
    [ u ] = msd_train_error( abs(H * BETA - T) );
    H = bsxfun(@times, u, H);
    T = bsxfun(@times, u, T);

    % ReCompute Beta: output weight
    if size(H,1) <= size(H,2)  % H is [N, K*(d-r+1)^2]
        BETA = H' * ((eye(N, N)/C + H*H') \ T); % A*inv(B)*C  --> A*(B\C)
    else
        BETA = (eye(size(H,2))/C +H'*H) \ (H'*T); % inv(A)*B  --> A\B
    end
end

train_time = toc;

Y = H*BETA;

% calculate training classification accuracy
[~, label0] = max(T, [], 2);
[~, label] = max(Y, [], 2);

bad = find(label0 ~= label);
er = numel(bad) / N;
train_accuracy = 1. - er

result = struct;
result.train_time = train_time;
result.train_accuracy = train_accuracy;
result.time_network = time_network;

param.W = W;
param.BETA = BETA;
param.pool_index = pool_index;

end

