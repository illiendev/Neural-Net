clc;
clear all; close all;

%% Initialization

numSamples = 2000;
image = randn(63, 63, 16);
kernel = single(randn(3, 3, 16, 11));
kernel1d = reshape(kernel, [], 11)';

[iCols, iRows, iDepth] = size(image);
[kCols, kRows, kDepth, numKernels] = size(kernel);

imageVector = single(reshape(1:(iCols*iRows*iDepth*numSamples), (iCols*iRows), iDepth, numSamples));
imageVector2d = reshape(imageVector, iCols, iRows, iDepth, numSamples);
imageVector3 = permute(imageVector, [2 1 3]);

[index, dindex] = build_index(image, kernel, [1 1], [1 1]);

images2d = randn(iCols, iRows, iDepth, numSamples, 'single');
images1d = reshape(permute(images2d, [3 1 2 4]), iDepth, [], numSamples);

imagesf = padarray(images2d, [1 1 1 0]) .* 0;
for i=1:numSamples
    imagesf(:,:,:,i) = fftn(padarray(images2d(:,:,:,i), [1 1 1]));
end

padval = (size(imagesf(:,:,:,1)) - size(kernel(:,:,:,1)))/2;

kernelsf = padarray(kernel, [padval 0]) .* 0;
for i=1:numKernels
    kernelsf(:,:,:,i) = fftn(padarray(kernel(:,:,:,i), padval));
end

lsize = numel(kernelsf(:,:,:,1));
imagesf = reshape(imagesf, lsize, []);
kernelsf = reshape(kernelsf, lsize, 1, 11);

index = uint32(index);
dindex = uint32(dindex);

f1 = @() imagesf .* kernelsf;
f2 = @() kernel1d * reshape(images1d(:, index, :), kCols*kRows*iDepth, []);

% x = timeit(f1);
% x2 = timeit(f2);
x = imagesf .* kernelsf;
z = kernel1d * reshape(images1d(:, index, :), kCols*kRows*iDepth, []);



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