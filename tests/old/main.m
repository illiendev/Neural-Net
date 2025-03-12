clc; addpath (pwd + "\data");
clear all; close all;

%% Hyperparameters (Split Network variables vs Data variables)
% Optimizer: 0- SGD, 1- SGDMOM, 2- ADAM
% Beta: Beta 1 and Beta 2 respectively
% Activation: 0- Linear, 1- ReLU, 2- Sigmoid, 3-Swish, 4-Softmax

% Possibility of creating a new class named layer that inherits attributes
% from either standard or conv layer. Difficulty would be in the
% conditional inheritance

[tdata, tlabel, vdata, vlabel] = mnist.load_data();
config.numEpochs = 1000;
config.maxTime = 600;
config.logInterval = 10;

config.optimizer = 2;
config.beta = [0.9 0.999];
config.structure = [50 50];
config.convStructure = {ones(3,3,5)};
config.convStride = {[1 1]};
config.activation = [1 4];

config.varType = single(1);
config.dataSize = size(tdata, 1);
config.labelSize = size(tlabel, 1);

N = 4;
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