clear all;clc;
filename = 'Nmaps16Fuzzy.mat';
logdir = './log/';
resultfile = [logdir, filename];
load(resultfile);

NCs = length(result_train);
Cs = zeros(NCs, 1);
TrainTime = zeros(NCs, 1);
TrainAcc = zeros(NCs, 1);
for n = 1:NCs
    Cs(n) = result_train{n}.C;
    TrainTime(n) = result_train{n}.train_accuracy;
    TrainAcc(n) = result_train{n}.train_time;
end


NCs = length(result_test);
TestTime = zeros(NCs, 1);
TestAcc = zeros(NCs, 1);
for n = 1:NCs
    TestTime(n) = result_test{n}.test_time;
    TestAcc(n) = result_test{n}.test_accuracy;
end


figure
subplot(121)
plot(Cs, TrainAcc);
grid on
xlabel('Balanced factor C');
ylabel('Accuracy');
title('Training accuracy on NORB dataset')
subplot(122)
plot(Cs, TestAcc);
grid on
xlabel('Balanced factor C');
ylabel('Accuracy');
title('Testing accuracy on NORB dataset')
