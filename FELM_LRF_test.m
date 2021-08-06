function [result] = FELM_LRF_test(X, T, param)
%FELM_LRF_TEST Summary of this function goes here
%   Detailed explanation goes here

[nX, hX, wX, cX] = size(X);

[nTest, nClass] = size(T);

X = reshape(X, nTest, hX*wX*cX);


tic;
layer_param = param.network_params{1};
[dummy, H] = two_layer_forwardprop(X, param.W, param.pool_index, layer_param.l1_act, layer_param.l2_act); 
time_network = toc;

Y = H*param.BETA;

test_time = toc;

N = nTest;
% calculate training classification accuracy
[~, label0] = max(T, [], 2);
[~, label] = max(Y, [], 2);

bad = find(label0 ~= label);
er = numel(bad) / N;
test_accuracy = 1. - er

result = struct;
result.test_time = test_time;
result.test_accuracy = test_accuracy;
result.time_network = time_network;
result.cf = cfmatrix(label0, label);
end

