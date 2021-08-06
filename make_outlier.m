function [y] = make_outlier(y, rate)
%MAKE_OUTLIER Summary of this function goes here
%   Detailed explanation goes here

nC = numel(unique(y));
N = length(y);
num = zeros(nC, 1);
for i = 1:nC
    num(i) = uint32(rate*sum(y == i));
end

rand('seed', 1)

for i = 1:nC
    idx = randperm(N, num(i));
    yy = y(idx);
    y(idx) = yy(randperm(num(i), num(i)));
end

end

