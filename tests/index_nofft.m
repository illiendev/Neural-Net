clc;
clear all; close all;

%% Initialization
type = (single(1));

numSamples = 100;
image = randn(9, 9, 1);
kernel = randn(3, 3, 2);

[iw, ih, depth] = size(image);
[kw, kh, numKernels] = size(kernel);
imageVector = single(reshape(1:(iw*ih*depth*numSamples), (iw*ih*depth), numSamples));
imageVector2 = double(imageVector)';

%% Index Initialization
[index, wIndex] = build_index(image, kernel, [1 1], [0 0]);
index = (index);
k1 = single(reshape(kernel, kw*kh, [])');
k2 = sparse(wIndex);
wIndex = (wIndex ~= 0);


te = full(reshape(k2(wIndex), 9, []));

x = k1 * reshape(imageVector(index, :), kw*kh, []);

y = imageVector2 * k2;

function [index, wIndex] = build_index(image, kernel, stride, padding)
rowStride = stride(1);
colStride = stride(2);
[imageCols, imageRows, depth] = size(image);
[kernelCols, kernelRows, numKernels] = size(kernel);
imageSize = imageCols*imageRows;
kernelSize = kernelCols*kernelRows;

imageIndex = padarray(reshape(1:imageSize, imageCols, imageRows), padding, 0);
[imageCols, imageRows] = size(imageIndex);
numCols = (1 + imageCols - kernelCols);
numRows = (1 + imageRows - kernelRows);

i = 1;
for row=1:rowStride:numRows
    for col=1:colStride:numCols
        index(:,i) = reshape(imageIndex(col:col+kernelCols-1, row:row+kernelRows-1), kernelSize, 1);
        i = i+1; 
    end
end

mapSize = size(index, 2);

% x = [];
% for i=1:depth
%     x = [x; index + imageSize*(i-1)];
% end
% index = x;

%% Weight Index
wIndex = zeros(imageSize, numKernels * prod(mapSize));
kernel = reshape(kernel, [], numKernels);
N = 1;
for j=1:numKernels
    for i=1:prod(mapSize)
        wIndex(index(:,i), N) = kernel(:, j);
        N = N+1;
    end
end
end