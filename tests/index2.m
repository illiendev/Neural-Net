clc;
clear all; close all;

%% Initialization

numSamples = 100;
image = randn(64, 64, 16);
kernel = randn(3, 3, 32);

[iCols, iRows, iDepth] = size(image);
[kCols, kRows, kDepth] = size(kernel);
imageVector = single(reshape(1:(iCols*iRows*iDepth*numSamples), (iCols*iRows), iDepth, numSamples));
imageVector2 = reshape(imageVector, (iCols*iRows*iDepth), numSamples);
imageVector3 = permute(imageVector, [2 1 3]);
[index, dindex] = build_index(image, kernel, [1 1], [1 1]);

index = uint32(index);
dindex = uint32(dindex);
% w = randn()

x = reshape(pagetranspose(imageVector(index, :, :)), kCols*kRows*iDepth, []);
y = reshape(imageVector2(dindex, :), kCols*kRows*iDepth, []);
z = reshape(imageVector3(:, index, :), kCols*kRows*iDepth, []);

% tic
% x1 = imageVector(index, :, :);
% x2 = pagetranspose(x1);
% x3 = reshape(x2, kCols*kRows*iDepth, []);
% toc

% tic
% imageVector = reshape(imageVector, (iCols*iRows*iDepth), numSamples);
% y1 = imageVector(dindex, :);
% y2 = reshape(y1, kCols*kRows*iDepth, []);
% toc
% 
% tic
% z1 = imageVector3(:, index, :);
% z2 = reshape(z1, kCols*kRows*iDepth, []);
% toc
% 
% y = pagetranspose(x);
% y = reshape(x, 27, 81, []);

function [index, dindex] = build_index(image, kernel, stride, padding)
[iCols, iRows, iDepth] = size(image);
[kCols, kRows, kDepth] = size(kernel);
rowStride = stride(1);
colStride = stride(2);

iSize = iCols*iRows;
kSize = kCols*kRows;

imageIndex = padarray(reshape(1:iSize, iCols, iRows), padding, 'symmetric');
[iCols, iRows] = size(imageIndex);
numCols = (1 + iCols - kCols);
numRows = (1 + iRows - kRows);

i = 1;
for row=1:rowStride:numRows
    for col=1:colStride:numCols
        index(:,i) = reshape(imageIndex(col:col+kCols-1, row:row+kRows-1), kSize, 1);
        i = i+1;
    end
end

x = [];
for i=1:iDepth
    x = [x; index + iSize*(i-1)];
end
dindex = x;

%% Weight Index
% wIndex = zeros(iSize, kDepth * prod(mapSize));
% kernel = reshape(kernel, [], kDepth);
% N = 1;
% for j=1:kDepth
%     for i=1:prod(mapSize)
%         wIndex(index(:,i), N) = kernel(:, j);
%         N = N+1;
%     end
% end
end