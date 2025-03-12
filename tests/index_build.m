clc;
clear all; close all;

%% Initialization
% load('data\numbers_2d.mat');
numKernels = 5;
numSamples = 100;
type = (single(1));

imageSize = [32 32];
kernelSize = [3 3];

data = imresize(data, imageSize);
image = data(:,:,1:numSamples);
kernel = randn(3,3,numKernels);

image_2d = type * imresize(image(:,:,1:numSamples), imageSize);
kernel_2d = type * imresize(kernel(:,:,1:numKernels), kernelSize);
image = type * reshape(image_2d, prod(imageSize), numSamples);

kernel = type * reshape(kernel_2d, prod(kernelSize), numKernels);

for i=1:numKernels
kernel_fft(:,i) = padarray(kernel(:,i), ceil((prod(imageSize)-prod(kernelSize))/2));
end
kernel_fft = imresize(kernel_fft, [prod(imageSize) numKernels]);
kernel_fft = type * fft(kernel_fft);
image_fft = type * fft(image);

kern_size = size(kernel_2d, 1);
imag_size = size(image_2d, 1);
numSlices = (imag_size - kern_size + 1)^2;


%% Index Initialization
index = build_index(size(image_2d(:,:,1)), size(kernel_2d(:,:,1)), 1, 1) * type;

for i=1:numKernels
    kernels(i).w = kernel_2d(:,:,i);
end

%% Reference
% ref = zeros(numSlices, numKernels, numSamples) * type;
tic;
% for i=1:numKernels
% ref(:,i,:) = default_conv(image_2d, kernel_2d(:,:,i), numSlices, numSamples);
% end
ref = arrayfun(@(x) default_conv(image_2d, x.w), kernels, 'UniformOutput', false);
tr = toc;

%% a
tic
a1 = image(index, :);
a2 = reshape(a1, length(kernel(:,1)), numSamples*numSlices);
a3 = reshape(kernel' * a2, [numKernels numSlices numSamples]);
% a5 = (reshape(permute(a4, [2 1 3]), numSlices, numKernels*numSamples));
% a = reshape(kernel' * reshape(image(index, :), length(kernel(:,1)), numSamples*numSlices), [numKernels*numSlices numSamples]);
ta = toc;

%% b
% w2 = [];
% for i=1:numKernels
% w2 = [w2 build_windex(size(image_2d(:,:,1)), size(kernel_2d(:,:,1)), 1, 1) * type];
% end
% w2 = single(w2);
% tic;
% b = reshape(w2' * image, [numKernels numSlices numSamples]);
% tb = toc;

%% c
c = type * zeros([size(image) numKernels]);
tic;
for i=1:numKernels
c(:, :, i) = ifft(image_fft .* kernel_fft(:, i));
end
tc = toc;

%% Results
fprintf('%f\n%f\n%f\n', ta, tr, ta/tr);
% imshow(reshape(a3, sqrt(size(a3,2)), []))
% figure(2)
% imshow(reshape(c, imageSize))

function result = default_conv(image_2d, kernel_2d)
%     result = reshape(convn(image_2d, kernel_2d, 'valid'), numSlices, numSamples);
    result = convn(image_2d, kernel_2d, 'valid');
end

function index = build_index(imageSize, kernelSize, rowStride, colStride)
image = reshape(1:prod(imageSize), imageSize);
kernel = ones(kernelSize);

numRows = (1 + size(image,1) - size(kernel,1));
numCols = (1 + size(image,2) - size(kernel,2));
numSlices = numCols * numRows;

x = 1;
for row=1:rowStride:numRows
    for col=1:colStride:numCols
        index(:,x) = reshape(image(col:col+size(kernel,1)-1, row:row+size(kernel,2)-1), length(kernel(:)), 1);   
        x = x+1;
    end
end
end