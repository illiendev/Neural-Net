classdef neural_net < handle
    properties
        layer, numLayers;
        convLayer, numConvLayers;
        emptyOutput;
    end

    methods
        function obj = neural_net(config)
            if isempty(config.convStructure), convLayer = []; end
            config.structure = ([config.structure config.labelSize]);
            numLayers = length(config.structure);
            numConvLayers = length(config.convStructure);

            %% Configure function index
            if length(config.activation) <= 2
                funcIndex = config.activation(1) .* ones((numLayers+numConvLayers), 1);
                funcIndex(end) = config.activation(end);
            else
                funcIndex = config.activation;
                if length(config.activation) ~= (numLayers+numConvLayers)
                    fprintf('Bad configuration. (Forgot output or conv layer?)\n');
                    pause(1);
                end
            end

            %% Initialize weights and biases
            L = 1;
            layerSize = round([sqrt(config.dataSize) sqrt(config.dataSize) 1]);
            for i=1:numConvLayers
                convLayer(i) = convolutional_layer(config, layerSize, i, funcIndex(L), numConvLayers);
                layerSize = [convLayer(i).sz.mcols, convLayer(i).sz.mrows, convLayer(i).sz.kdepth];
                emptyOutput(L).a = convLayer(L).bn;
                L = L+1;
            end

            if isempty(convLayer), layerSize = config.dataSize; convLayer = [];
            else, layerSize = prod([convLayer(end).sz.map convLayer(end).sz.kdepth]); end

            config.structure = [layerSize config.structure];
            for i=1:numLayers
                layer(i) = standard_layer(config, i, funcIndex(L), numLayers);
                emptyOutput(L).a = layer(i).bn;
                L = L+1;
            end

            %% Initialize constructor
            obj.emptyOutput = emptyOutput;
            obj.layer = layer;
            obj.convLayer = convLayer;
            obj.numLayers = numLayers;
            obj.numConvLayers = numConvLayers;
        end

        %% Backpropagation
        function backprop(obj, tdata, tlabel, rate, batch_size, epoch)
            numSamples = size(tdata, 2);
            randIndex = single(randperm(numSamples));
            tdata = tdata(:, randIndex);
            tlabel = tlabel(:, randIndex);
            mainIndex = single(0:batch_size:numSamples);
            mainIndex(end) = numSamples;

            for i=1:length(mainIndex) - 1
                p1 = mainIndex(i);
                p2 = mainIndex(i+1);
                obj.calc_gradient(tdata(:, (1+p1):p2), tlabel(:, (1+p1):p2), rate, epoch);
            end
        end

        %% Feedforward
        function [output, accuracy] = feed_forward(obj, data, label)
            L = 1; output = obj.emptyOutput;

            for i=1:obj.numConvLayers
                if L == 1, output(1).a = obj.convLayer(1).forward_pass(data);
                else, output(L).a = obj.convLayer(i).forward_pass(output(L-1).a); end
                L = L+1;
            end

            for i=1:obj.numLayers
                if L == 1, output(1).a = obj.layer(1).forward_pass(data);
                else, output(L).a = obj.layer(i).forward_pass(output(L-1).a); end
                L = L+1;
            end

            [~, ix] = max(output(end).a);
            [~, iy] = max(label);
            accuracy = mean(ix == iy);
        end

        %% Gradient Calculation
        function calc_gradient(obj, tdata, tlabel, rate, epoch)
            L = obj.numLayers + obj.numConvLayers;
            batch_size = single(size(tlabel, 2));
            lrate = rate / (batch_size);
            output = obj.feed_forward(tdata, tlabel);
            dout = (output(end).a - tlabel);

            for i=obj.numLayers:-1:1
                if L ~= 1
    				dout = obj.layer(i).backward_pass(dout, output(L-1).a, output(L).a, batch_size, epoch, lrate);
                else
    				obj.layer(i).backward_pass(dout, tdata, output(L).a, batch_size, epoch, lrate);
                end
                L = L-1;
            end

            for i=obj.numConvLayers:-1:1
                if L ~= 1
    				dout = obj.convLayer(i).backward_pass(dout, output(L).a, batch_size, epoch, lrate);
                else
    				obj.convLayer(i).backward_pass(dout, tdata, output(L).a, batch_size, epoch, lrate);
                end
                L = L-1;
            end
        end

        %% Validation
        function [accuracy] = validate(obj, data, label)
            [~, accuracy] = obj.feed_forward(data, label);
        end

        %% Network Training
        function [performance] = train_network(obj, config, tdata, tlabel, vdata, vlabel)
            performance = []; time = 0;
            for epoch=1:config.numEpochs
                tic;
                if (epoch == 1), batch_size = 100; rate = single(1 * 1E-3); end

                % sqrt(batchsize)
                % changing beta
%                 obj.backprop(mnist.new_sample(tdata), tlabel, rate, batch_size, epoch);
                obj.backprop(tdata, tlabel, rate, batch_size, epoch);
                time = time + toc;

                if mod(epoch, config.logInterval) == 0
                    performance(end+1, :) = [obj.validate(vdata, vlabel), obj.validate(tdata, tlabel), time];
                    clc;
                    fprintf ('Epoch(%d): %fs (%fs)', epoch, time, time/epoch);
                    fprintf ('\nAccuracy (Training):\t%f', performance(end, 2));
                    fprintf ('\nAccuracy (Validation):\t%f\n---------------\n', performance(end, 1));
                    if performance(end, 2) >= .9987, break; end
                end

%                 if epoch == 1, fprintf("First run: %fs\n", time); end
                if time > config.maxTime, break; end
            end
        end
    end
end