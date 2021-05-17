classdef neural_net
properties
layer, numLayers;
emptyOutput, emptyGradient;
m, v, b1, b2, bias_fix;
params;
funcIndex;
train;
end

methods
%% Neural Network constructor
function obj = neural_net(config)
    cpuVar = (single(1));
    obj.numLayers = length(config.structure) - 1;
    obj.layer = struct('bn', cpuVar*0, 'wn', cpuVar*0, 'w', num2cell(cpuVar*(zeros(1, obj.numLayers))));
    obj.emptyOutput = struct('a', num2cell(cpuVar*(zeros(1, obj.numLayers))));
    obj.emptyGradient = obj.layer;

    for i=1:obj.numLayers
        obj.layer(i).w = cpuVar * (randn(config.structure(i+1), config.structure(i))/sqrt(config.structure(i)));
        obj.layer(i).wn = cpuVar * (ones(config.structure(i+1), 1));
        obj.layer(i).bn = cpuVar * (zeros(config.structure(i+1), 1));
        obj.emptyGradient(i).w = cpuVar * (zeros(config.structure(i+1), config.structure(i)));
        obj.emptyGradient(i).wn = cpuVar * (zeros(config.structure(i+1), 1));
        obj.emptyGradient(i).bn = cpuVar * (zeros(config.structure(i+1), 1));
        obj.emptyGradient(i).da = cpuVar * (zeros(config.structure(i+1), 1));
        obj.emptyGradient(i).dz = cpuVar * (zeros(config.structure(i+1), 1));
        obj.emptyOutput(i).a = cpuVar * (zeros(config.structure(i+1), 1));
    end

    %% Momentum
    obj.params.optimizer = config.optimizer;
    epochs = (1:config.numEpochs);
    obj.b1 = config.beta(1);
    obj.b2 = config.beta(2);
    % Possible alternative to bias fix. Instead of having to fix the
    % bias because it is initialized to zero, have both m and v be
    % gradually initialized from raw SGD to momentum based.
    %         obj.b1 = config.b1 - config.b1 * exp(-epochs/10);
    %         obj.b2 = config.b2 - config.b2 * exp(-epochs/10);
    obj.m = obj.emptyGradient;
    obj.v = obj.emptyGradient;
    obj.bias_fix = single(sqrt(1-obj.b2.^epochs) ./ (1 - obj.b1.^epochs));

    if length(config.activation) <= 2
        obj.funcIndex = ones(obj.numLayers, 1);
        obj.funcIndex(1) = config.activation(1);
        obj.funcIndex(end) = config.activation(end);
    else
        obj.funcIndex = config.activation;
        if length(config.activation) ~= obj.numLayers
            fprintf('Bad configuration. (Forgot output layer?)\n');
            pause(1);
        end
    end
end

%% Feed Forward
function [output, accuracy, cache] = feed_forward(obj, data, label)
    output = obj.emptyOutput;
    [output(1).a, cache(1)] = obj.activation(obj.layer(1), data, obj.funcIndex(1));
    for i=2:obj.numLayers
        [output(i).a, cache(i)] = obj.activation(obj.layer(i), output(i-1).a, obj.funcIndex(i));
    end
    [~, ix] = max(output(end).a);
    [~, iy] = max(label);
    accuracy = mean(ix == iy);
end

%% Backpropagation
function [obj] = backprop(obj, tdata, tlabel, rate, batch_size, epoch)
    obj.train = 1;
    numSamples = size(tdata, 2);
    randIndex = single(randperm(numSamples));
    tdata = tdata(:, randIndex);
    tlabel = tlabel(:, randIndex);
    mainIndex = single(0:batch_size:numSamples);
    mainIndex(end) = numSamples;
    for i=1:length(mainIndex) - 1
        p1 = mainIndex(i); p2 = mainIndex(i+1); lrate = rate / (p2 - p1);
        [gradient] = obj.calc_gradient(tdata(:, (1+p1):p2), tlabel(:, (1+p1):p2));
        switch obj.params.optimizer
            case 0
                [obj] = obj.sgd(gradient, lrate);
            case 1
                [obj] = obj.sgd_mom(obj.calc_gradient(tdata(:, (1+p1):p2), tlabel(:, (1+p1):p2)), lrate);
            case 2
                [obj] = obj.adam(obj.calc_gradient(tdata(:, (1+p1):p2), tlabel(:, (1+p1):p2)), lrate, epoch);
        end
    end

    obj.train = 0;
end

%% Gradient Calculation
function [gradient] = calc_gradient(obj, tdata, tlabel)
    gradient = obj.emptyGradient;
    [output, ~, cache] = feed_forward(obj, tdata, tlabel);
    N = size(tdata, 2);

    for L=obj.numLayers:-1:1
        if L==obj.numLayers
            gradient(L).da = (output(L).a - tlabel);
        else, gradient(L).da = (obj.layer(L+1).w' * gradient(L+1).dz) .* obj.activation_prime(output(L).a, obj.funcIndex(L));
        end
        gradient(L).bn = sum(gradient(L).da, 2);
        gradient(L).wn = gradient(L).bn .* obj.layer(L).wn;
        dzhat = gradient(L).da .* obj.layer(L).wn;
        gradient(L).dz = 1 ./ (N .* cache(L).var) .* (N.*dzhat - sum(dzhat, 2) - cache(L).zhat .* sum(dzhat .* cache(L).zhat, 2));
        if L==1
            gradient(L).w = gradient(L).dz * tdata';
        else, gradient(L).w = gradient(L).dz * output(L-1).a';
        end
    end
end

%% Optimizers
function [obj] = sgd(obj, gradient, lrate)
    for L=1:obj.numLayers
        obj.layer(L).w = obj.layer(L).w - lrate * gradient(L).w;
        obj.layer(L).wn = obj.layer(L).wn - lrate * gradient(L).wn;
        obj.layer(L).bn = obj.layer(L).bn - lrate * gradient(L).bn;
    end
end

function [obj] = sgd_mom(obj, gradient, lrate)
    for L=1:obj.numLayers
        obj.m(L).bn = (obj.b1 * obj.m(L).bn + lrate * (1-obj.b1) * gradient(L).bn);
        obj.m(L).wn = (obj.b1 * obj.m(L).wn + lrate * (1-obj.b1) * gradient(L).wn);
        obj.m(L).w = (obj.b1 * obj.m(L).w + lrate * (1-obj.b1) * gradient(L).w);
        obj.layer(L).bn = obj.layer(L).bn - obj.m(L).bn;
        obj.layer(L).wn = obj.layer(L).wn - obj.m(L).wn;
        obj.layer(L).w = obj.layer(L).w - obj.m(L).w;
    end
end

function [obj] = adam(obj, gradient, lrate, epoch)
    for L=1:obj.numLayers
        obj.m(L).bn = (obj.b1 * obj.m(L).bn + lrate * (1-obj.b1) * gradient(L).bn);
        obj.m(L).wn = (obj.b1 * obj.m(L).wn + lrate * (1-obj.b1) * gradient(L).wn);
        obj.m(L).w = (obj.b1 * obj.m(L).w + lrate * (1-obj.b1) * gradient(L).w);
        obj.v(L).bn = (obj.b2 * obj.v(L).bn + lrate * (1-obj.b2) * (gradient(L).bn).^2);
        obj.v(L).wn = (obj.b2 * obj.v(L).wn + lrate * (1-obj.b2) * (gradient(L).wn).^2);
        obj.v(L).w = (obj.b2 * obj.v(L).w + lrate * (1-obj.b2) * (gradient(L).w).^2);
        obj.layer(L).bn = obj.layer(L).bn - obj.bias_fix(epoch) .* obj.m(L).bn ./ sqrt(obj.v(L).bn) + eps(single(1));
        obj.layer(L).wn = obj.layer(L).wn - obj.bias_fix(epoch) .* obj.m(L).wn ./ sqrt(obj.v(L).wn) + eps(single(1));
        obj.layer(L).w = obj.layer(L).w - obj.bias_fix(epoch) .* obj.m(L).w ./ sqrt(obj.v(L).w) + eps(single(1));
    end
end

%% Validation
function accuracy = validate(obj, data, label)
    result = feed_forward(obj, data, label);
    [~, ix] = max(result(end).a);
    [~, iy] = max(label);
    accuracy = mean(ix == iy);
end

%% Utility
function surf(obj, layer)
    switch nargin
        case 1
            for i=1:obj.numLayers
                subplot(obj.numLayers,1,i)
                surf(obj.layer(i).w, 'EdgeColor', 'none');
            end
        case 2
            for i=1:obj.numLayers
                subplot(obj.numLayers,1,i)
                surf(layer(i).da, 'EdgeColor', 'none');
            end
    end
end
end

methods (Static)
function [output, cache] = batch_norm(input, layer)
    dim = size(input, 2);
    cache.avg = sum(input, 2) / dim;
    cache.var = sum((input - cache.avg).^2, 2) / (dim-1);
    cache.zhat = (input - cache.avg) ./ sqrt(cache.var + eps(single(1)));
    output = cache.zhat .* layer.wn + layer.bn;
end

function [a, cache] = activation(layer, data, type)
    z = layer.w * data;
    [z, cache] = neural_net.batch_norm(z, layer);
    switch type
        case 0
            a = z; % Linear
        case 1
            a = max(0, z); % Relu
        case 2
            a = 1./(1 + exp(-z)); % Sigmoid
        case 3
            a = z./(1 + exp(-z)); % Swish
        case 4
            a = exp(z)./sum(exp(z)); % Softmax
    end
end

function da = activation_prime(a, type)
    switch type
        case 0
            da = ones(size(a)); % Linear
        case 1
            da = single(a>0); % Relu
        case 2
            da = a .* (1 - a); % Sigmoid
        case 3
            da = (1.421 .* a + 0.4401) ./ (a + 0.8752); % Swish approximation (assumes a domain = [-1 1])
        case 4
            da = a .* (1 - a); % Softmax
    end
end
end
end