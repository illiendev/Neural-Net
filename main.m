clc;
clear all; close all;

%% Hyperparameters
config.numEpochs = 1000;
config.maxTime = 500;
config.logInterval = 10;

config.optimizer = 0;           % 0-SGD, 1-SGDMOM, 2-ADAM
config.beta = [0.9 0.999];      % beta1 and beta2 respectively
config.structure = [100];
config.activation = [1 4];
[net, perf] = train_network(config, 'data\mnist_uint8.mat');

function [net, performance] = train_network(config, location)
[tdata, tlabel, vdata, vlabel] = data_processing(location);
config.structure = [size(tdata, 1) config.structure size(tlabel, 1)];
net = neural_net(config);
performance = []; time = 0;

for epoch=1:config.numEpochs
    tic;
    if (epoch == 1), batch_size = 100; rate = 4 * 1E-3; end
%     if (epoch == 100), batch_size = 100; rate = 1 * 1E-3; end
%     if (epoch == 200), batch_size = 50; rate = 1 * 1E-3; end
%     if (epoch == 300), batch_size = 50; rate = 1 * 1E-3; end

    net = net.backprop(tdata, tlabel, rate, batch_size, epoch);
    time = time + toc;
    
    if mod(epoch, config.logInterval) == 0
        performance(end+1, :) = [net.validate(vdata, vlabel), net.validate(tdata, tlabel), time];
        fprintf ('Epoch(%d): %fs', epoch, time);
        fprintf ('\nError(Training): %f', performance(end, 2));
        fprintf ('\nAccuracy: %f\n---------------\n', performance(end, 1));
        if performance(end, 2) >= 1, break; end
    end

    if time > config.maxTime, break; end
end
end

function [tdata, tlabel, vdata, vlabel] = data_processing(location)
load(location, 'tdata', 'tlabel', 'vdata', 'vlabel');
tdata = single(tdata)/255;
tlabel = single(tlabel);
vdata = single(vdata)/255;
vlabel = single(vlabel);
tdata = (tdata - mean(tdata(:)) ./ (std(tdata(:))));
vdata = (vdata - mean(vdata(:)) ./ (std(vdata(:))));
end
