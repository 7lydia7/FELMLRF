clear all;clc;
filename1 = 'Nmaps32Fuzzy.mat';
filename2 = 'Nmaps32FuzzyClassDist.mat';
logdir = './log/';
resultfile1 = [logdir, filename1];
resultfile2 = [logdir, filename2];

load(resultfile1);
NCs = length(result_train)-1;
Cs = zeros(NCs, 1);
TrainTime = zeros(NCs, 1);
TrainAcc = zeros(NCs, 1);
for n = 1:NCs
    Cs(n) = result_train{n}.C;
    TrainTime(n) = result_train{n}.train_time;
    TrainAcc(n) = result_train{n}.train_accuracy;
end

NCs = length(result_test)-1;
TestTime = zeros(NCs, 1);
TestAcc = zeros(NCs, 1);
for n = 1:NCs
    TestTime(n) = result_test{n}.test_time;
    TestAcc(n) = result_test{n}.test_accuracy;
end


load(resultfile2);
NCs = length(result_train)-1;
Cs = zeros(NCs, 1);
TrainTimeFuzzy = zeros(NCs, 1);
TrainAccFuzzy = zeros(NCs, 1);
for n = 1:NCs
    Cs(n) = result_train{n}.C;
    TrainTimeFuzzy(n) = result_train{n}.train_time;
    TrainAccFuzzy(n) = result_train{n}.train_accuracy;
end

NCs = length(result_test)-1;
TestTimeFuzzy = zeros(NCs, 1);
TestAccFuzzy = zeros(NCs, 1);
for n = 1:NCs
    TestTimeFuzzy(n) = result_test{n}.test_time;
    TestAccFuzzy(n) = result_test{n}.test_accuracy;
end

figure(1)
grid on
hold on
plot(Cs, TrainAcc, '-b+', 'LineWidth', 2);
plot(Cs, TrainAccFuzzy, '-ro', 'LineWidth', 2);
legend({'ELM-LRF', 'FELM-LRF'});
xlabel('Balanced factor C');
ylabel('Accuracy');
title('Training accuracy on NORB dataset')

figure(2)
grid on
hold on
plot(Cs, TestAcc, '-b+', 'LineWidth', 2);
plot(Cs, TestAccFuzzy, '-ro', 'LineWidth', 2);
legend({'ELM-LRF', 'FELM-LRF'});
xlabel('Balanced factor C');
ylabel('Accuracy');
title('Testing accuracy on NORB dataset')
