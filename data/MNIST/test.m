% load('data\MNIST\mnist_uint8.mat', 'tdata', 'tlabel', 'vdata', 'vlabel');
clc;
clear all; close all;

% "t10k-images.idx3-ubyte" "t10k-labels.idx1-ubyte"
% "train-images.idx3-ubyte" "train-labels.idx1-ubyte"
filename = "train-images.idx3-ubyte";
images =  uint32(loadMNISTImages(filename));
images2 =  uint32(fread(fopen(filename, 'rb'), inf, 'unsigned char'));
images3 = uint32(fread(fopen(filename)));

x = sum(images3~=images2);
y = sum(images~=(images3(17:end)));

function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
% images = reshape(images, numCols, numRows, numImages);
% images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
% images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
% images = double(images) / 255;

end