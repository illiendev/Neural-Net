clc; addpath (pwd + "\data");
clear all; close all;

%% Hyperparameters (Split Network variables vs Data variables)
% Optimizer: 0- SGD, 1- SGDMOM, 2- ADAM
% Beta: Beta 1 and Beta 2 respectively
% Activation: 0- Linear, 1- ReLU, 2- Sigmoid, 3-Swish, 4-Softmax

[tdata, tlabel, vdata, vlabel] = mnist.load_data();
config.numEpochs = 500;
config.maxTime = 3600;
config.logInterval = 10;

config.optimizer = 2;
config.beta = [0.9 0.999];
config.structure = [100 100];
config.convStructure = {ones(3,3,9)};
config.convStride = {[1 1]};
config.activation = [1 4];

config.varType = single(1);
config.dataSize = size(tdata, 1);
config.labelSize = size(tlabel, 1);

N = 1;
results = train_batch(N, config, tdata, tlabel, vdata, vlabel);

function results = train_batch(N, config, tdata, tlabel, vdata, vlabel)
results = [];
net(1:N) = neural_net(config);
for i=1:N
    net(i) = neural_net(config);
    performance = net(i).train_network(config, tdata, tlabel, vdata, vlabel);
    results = [results; max(performance) .* [100 100 1]];
end
% results = performance;
end