clc;
clear all; close all;

image = ones(9,9);
kernel = ones(3,3);

index = buildIndex(image, kernel, [0 0]);

function [index, mapSize] = buildIndex(image, kernel, stride)
% random stride, build multiple indexes and have a chance of using stride A
% or B (iteration 1 would be 1 and 3, iteration 2 2 and 4)
% rowStride = stride(1);
% colStride = stride(2);

[iCols, iRows, iDepth] = size(image);
[kCols, kRows, kDepth] = size(kernel);

% padding = floor([kCols, kRows]/2);
padding = [0 0];

iSize = iCols*iRows;
kSize = kCols*kRows;

imageIndex = padarray(reshape(1:iSize, iCols, iRows), padding, 'symmetric');
[iCols, iRows] = size(imageIndex);
numCols = (1 + iCols - kCols);
numRows = (1 + iRows - kRows);
mapSize = [numCols, numRows];

index = zeros(iSize, numCols*numRows);
i = 1;
for row=1:numRows
    for col=1:numCols
        index(reshape(imageIndex(col:col+kCols-1, row:row+kRows-1), kSize, 1), i) = 1:kSize;
        i = i+1;
    end
end
end