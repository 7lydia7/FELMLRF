function [ u, label ] = msd_class_dist( x, y )
%CLASS_CENTER Summary of this function goes here
%   Detailed explanation goes here
% x: H*W*C-N
% y: 1-N or nC-N

[nX, hX, wX, cX] = size(x);

x = reshape(x, nX, hX*wX*cX).';

if ~isvector(y) % one hot --> normal
    [~, y] = max(y, [], 2);
end

[n, N] = size(x);
label = unique(y);
nC = numel(label);
y = y - min(label) + 1;
newlabel = 1:nC;

s = zeros(n, nC);
num = zeros(1, nC);

for i = 1:N
    k = y(i);
    num(k) = num(k) + 1;
    s(:, k) = s(:, k) + x(:, i);
end
% 
% [C, U] = fcm(x', 10, [2.0, 1000, 1e-8, 1]);
% 
% [U, centers] = my_fuzzyCmeans(x', 10);

% [IDX, C] = kmeans(x', 10);

s = bsxfun(@rdivide, s, num);

d = zeros(N, 1);
for i = 1:N
    d(i) = norm(x(:, i) - s(:, y(i)));
end

r = zeros(nC, 1);
for k = newlabel
    r(k) = max(d(y==k));
end
delta = 1.0e-16;
r = r + delta;
u = 1 - d./r(y);
% u(u > 0.2) = 1.0;
% u(u <= 0.2) = 0.0;

% thd = mean(d);
% idxgt = d > thd;
% idxlt = d <= thd;
% u = zeros(1, N);
% u(idxgt) = 1 - d(idxgt)./r(y(idxgt));
% u(idxlt) = d(idxlt)/r(y(idxlt));
% % u(idxlt) = 1.0;












