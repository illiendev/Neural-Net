clc;
clear all; close all;


[vdata, a] = loadMNISTImages('t10k-images-idx3-ubyte');
labelv = loadMNISTLabels('t10k-labels-idx1-ubyte');
[tdata, b] = loadMNISTImages('train-images-idx3-ubyte');
labelt = loadMNISTLabels('train-labels-idx1-ubyte');


tlabel = uint8(zeros(10, 60000));
for i=1:60000
    temp = zeros(10,1);
    temp(labelt(i)+1) = 1;
    tlabel(:,i) = temp;
end

vlabel = uint8(zeros(10, 10000));
for i=1:10000
    temp = zeros(10,1);
    temp(labelv(i)+1) = 1;
    vlabel(:,i) = temp;
end