clear all;
clc;

rate = 0.01;

%% load NORB data
disk = 'D:/';
disk = '/mnt/d/';
% for training
load([disk, '/DataSets/oi/nsi/NORB/norb_traindata.mat']); %X is H*W*C-N, Y is N-1
train_x = reshape(X', size(X,2), 32, 32, 2); % X is H*W*C-N --> N-H-W-C
YY = make_outlier(Y, rate);
rate = sum(Y ~= YY) / length(Y)
Y = YY;
train_y = full(sparse(1:size(Y, 1), Y, 1)); % Y is N-1  -->  N*nClasses
% train_y(train_y==0) = -1;
% for testing
load([disk, '/DataSets/oi/nsi/NORB/norb_testdata.mat']);
test_x = reshape(X', size(X,2), 32, 32, 2);
test_y = full(sparse(1:size(Y, 1), Y, 1));
% test_y(test_y==0) = -1;
clear X Y YY;

startup;

%% set params
param_file = 'param32';
load(param_file);

num_maps = param.network_params{1}.num_maps;
disp(['num_maps: ', num2str(num_maps)]);

[ u, label ] = msd_class_dist(train_x, train_y);

param.u = u;
param.isUseClassDistFuzzy = 0;
param.isUseTrainErrorFuzzy = 0;

disp(['isUseClassDistFuzzy: ', num2str(param.isUseClassDistFuzzy)]);
disp(['isUseTrainErrorFuzzy: ', num2str(param.isUseTrainErrorFuzzy)]);

Cs = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,  0.1, 0.2, 0.3, 0.4, 0.5 0.6 0.7 0.8 0.9 1];
cnt = 1;
for C = Cs
    disp(['--------------C=', num2str(C), '----------------']);
     [ result_train{cnt}, param ] = FELM_LRF_train(train_x, train_y, param, C);
     [ result_test{cnt} ] = FELM_LRF_test(test_x, test_y, param);
     result_train{cnt}.C = C;
     result_test{cnt}.C = C;
     cnt = cnt+1;
end

fuzzystr = '';
if param.isUseClassDistFuzzy > 0
    fuzzystr = [fuzzystr, 'ClassDist'];
end
if param.isUseTrainErrorFuzzy > 0
    fuzzystr = [fuzzystr, 'TrainError'];
end

savestr = ['./log/Nmaps', num2str(num_maps), 'Fuzzy', fuzzystr,  '.mat'];

save(savestr, 'result_train', 'result_test', 'rate', 'C')
