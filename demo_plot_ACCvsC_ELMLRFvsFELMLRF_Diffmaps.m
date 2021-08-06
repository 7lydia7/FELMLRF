clear all; clc

TestAcc{1} = [0.9147    0.9267    0.9252    0.9213    0.9187    0.9172    0.9168    0.9153    0.9136    0.9119    0.9107    0.9095    0.9017    0.8967    0.8935    0.8904    0.8877    0.8864    0.8842    0.8824];
TestAccFuzzy{1} = [0.8641    0.9024    0.9109    0.9163    0.9171    0.9172    0.9170    0.9167    0.9157    0.9151    0.9142    0.9140    0.9087    0.9037    0.9000    0.8977    0.8956    0.8935    0.8916    0.8901];
TrainAcc{1} = [0.9651    0.9787    0.9830    0.9860    0.9874    0.9881    0.9884    0.9889    0.9892    0.9896    0.9900    0.9903    0.9909    0.9912    0.9914    0.9914    0.9915    0.9916    0.9916    0.9917];
TrainAccFuzzy{1} = [0.9290    0.9596    0.9689    0.9765    0.9800    0.9819    0.9831    0.9840    0.9848    0.9856    0.9865    0.9868    0.9888    0.9897    0.9902    0.9907    0.9908    0.9910    0.9911    0.9913];
TrainTime{1} = [4.6897    4.9396    3.9256    4.0166    4.3257    4.2010    4.4701    5.4002    5.1863    4.9282    5.2918    5.1468    7.9091    8.1287    8.0200    7.9661    8.0556    8.6420    8.0911    7.5164];
TrainTimeFuzzy{1} = [7.4682    7.6341    7.3262    7.4296    7.4646    8.0865    7.8695    7.4908    7.5365    3.9156    5.6425    5.3081    5.6347    5.0979    4.6734    3.8510    4.3321    4.7459    3.9291    3.8374];



TestAcc{2} = [0.9414    0.9494    0.9483    0.9462    0.9440    0.9428    0.9421    0.9407    0.9394    0.9384    0.9375    0.9366    0.9307    0.9267    0.9242    0.9215    0.9199    0.9176    0.9163    0.9149];
TestAccFuzzy{2} = [0.9030    0.9376    0.9439    0.9454    0.9448    0.9442    0.9430    0.9422    0.9420    0.9415    0.9406    0.9393    0.9354    0.9315    0.9286    0.9267    0.9253    0.9235    0.9222    0.9212];
TrainAcc{2} = [0.9812    0.9892    0.9903    0.9911    0.9915    0.9917    0.9917    0.9918    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919];
TrainAccFuzzy{2} = [0.9557    0.9782    0.9840    0.9882    0.9894    0.9902    0.9905    0.9909    0.9911    0.9913    0.9914    0.9915    0.9919    0.9919    0.9920    0.9920    0.9920    0.9920    0.9921    0.9921];
TrainTime{2} = [24.2531   23.6884   23.9089   24.3059   24.5300   24.3117   24.6550   24.5363   24.1820   24.5182   24.3998   24.4798   24.5484   24.5741   24.8331   24.4017   24.6241   24.2294   24.4887   24.2665];
TrainTimeFuzzy{2} = [23.5986   23.1755   26.1881   24.8537   24.6064   24.3739   24.8712   24.7082   24.7738   24.7919   24.7900   24.6751   25.0667   24.6165   24.7329   25.1274   24.5686   24.7782   24.5333   24.9732];



TestAcc{3} = [0.9586    0.9617    0.9594    0.9554    0.9517    0.9502    0.9482    0.9474    0.9463    0.9449    0.9444    0.9435    0.9381    0.9340    0.9306    0.9279    0.9253    0.9228    0.9212    0.9191];
TestAccFuzzy{3} = [0.9369    0.9556    0.9585    0.9582    0.9574    0.9563    0.9553    0.9538    0.9527    0.9518    0.9512    0.9505    0.9451    0.9419    0.9397    0.9384    0.9367    0.9356    0.9342    0.9330];

TrainAcc{3} = [0.9876    0.9912    0.9918    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919    0.9919];
TrainAccFuzzy{3} = [0.9756    0.9874    0.9898    0.9911    0.9915    0.9918    0.9919    0.9919    0.9920    0.9920    0.9920    0.9920    0.9921    0.9921    0.9921    0.9921    0.9921    0.9921    0.9921    0.9921];

TrainTime{3} = [73.8873   86.5230   87.5168   87.3870   88.0361   87.3571   88.5385   87.7708   88.7250   88.7649   90.0390   90.3029   88.9967   90.6943   89.3217   88.8439   89.1183   88.8456   94.9828   92.2968];
TrainTimeFuzzy{3} = [87.3981   87.9061   88.1119   88.1587   88.0015   87.7535   87.6272   90.4462   89.8263   90.0875   90.7965   90.3097   90.3430   88.2390   90.0638   90.0191   90.0511   90.8682   89.4687   89.0161];

Cs = [0.0010    0.0050    0.0100    0.0200    0.0300    0.0400    0.0500    0.0600    0.0700    0.0800    0.0900 0.1000    0.2000    0.3000    0.4000    0.5000    0.6000    0.7000    0.8000    0.9000
];

Nmaps = [8 16 32];

NN = length(Nmaps);
legendstr = cell(1, NN*2);
for k = 1:NN
    legendstr{2*k-1} = ['ELM-LRF(K=', num2str(Nmaps(k)), ')'];
    legendstr{2*k} = ['FELM-LRF(K=', num2str(Nmaps(k)), ')'];
end

figure(1)
grid on
hold on
for k = 1:NN
    plot(Cs, TrainAcc{k}, '-b+', 'LineWidth', k);
    plot(Cs, TrainAccFuzzy{k}, '-ro', 'LineWidth', k);
end
legend(legendstr);
xlabel('Trade off factor \lambda');
ylabel('Accuracy');
title('Training accuracy on NORB dataset')

figure(2)
grid on
hold on
for k = 1:NN
    plot(Cs, TestAcc{k}, '-b+', 'LineWidth', k);
    plot(Cs, TestAccFuzzy{k}, '-ro', 'LineWidth', k);
end
legend(legendstr);
xlabel('Trade off factor \lambda');
ylabel('Accuracy');
title('Testing accuracy on NORB dataset')

figure(3)
grid on
hold on
for k = 1:NN
    plot(Cs, TrainTime{k}, '-b+', 'LineWidth', k);
    plot(Cs, TrainTimeFuzzy{k}, '-ro', 'LineWidth', k);
end
legend(legendstr);
xlabel('Trade off factor \lambda');
ylabel('Time/s');
title('Training time on NORB dataset')
