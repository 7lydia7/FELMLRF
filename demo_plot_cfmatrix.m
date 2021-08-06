
clear all;clc;
filename = 'Nmaps16Fuzzy.mat';
logdir = './log/';
resultfile = [logdir, filename];
load(resultfile);
classname = {'Animal', 'Human', 'Airplane', 'Truck', 'Car'};
cmap = 'summer';
n = 3;
cfmat = result_test{n}.cf;
vfmt = '%d'; % '%0.02f';
plot_cfmatrix(cfmat, classname, vfmt, cmap)

