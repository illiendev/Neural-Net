clc;
clear all; close all;

% load('data\MNIST\mnist_uint8.mat', 'tdata', 'tlabel', 'vdata', 'vlabel');
tdata = permute(loaddata("train-images.idx3-ubyte"), [2 1 3]);
vdata = permute(loaddata("t10k-images.idx3-ubyte"), [2 1 3]);

tlabel = uint8(zeros(10, 60000));
tlabel2 = loadlabel("train-labels.idx1-ubyte");

vlabel = uint8(zeros(10, 10000));
vlabel2 = loadlabel("t10k-labels.idx1-ubyte");

for i=1:60000
tlabel(tlabel2(i)+1, i) = 1;
end

for i=1:10000
vlabel(vlabel2(i)+1, i) = 1;
end

% for i=1:100
%     n = randi([1 10000]);
%     imshow(vdata(:,:,n));
%     vlabel2(n)
%     pause(1)
% end

tdata = reshape(tdata, [], 60000);
vdata = reshape(vdata, [], 10000);

function output = loaddata(filename)
output = uint8(fread(fopen(filename)));
output = reshape(output(17:end), 28, 28, []);
end

function output = loadlabel(filename)
output = uint8(fread(fopen(filename)));
output = output(9:end);
end