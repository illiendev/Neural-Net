classdef neural_net < handle
    properties
        handler, numLayers;
        emptyOutput;
    end

    methods
        function obj = neural_net(config)
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

            %% Initialize layers
            layerSize = round([sqrt(config.dataSize) sqrt(config.dataSize) 1]);
            L = 1;
            for i=1:numConvLayers
                handler(i) = layer_handler(config, layerSize, i, funcIndex(L), numConvLayers, 1);
                layerSize = [handler(i).layer.sz.mcols, handler(i).layer.sz.mrows, handler(i).layer.sz.kdepth];
                emptyOutput(L).a = handler(L).layer.bn;
                L = L+1;
            end

            if numConvLayers==0, layerSize = config.dataSize;
            else, layerSize = prod([handler(numConvLayers).layer.sz.map handler(numConvLayers).layer.sz.kdepth]); end

            config.structure = [layerSize config.structure];
            for i=1:numLayers
                handler(L) = layer_handler(config, layerSize, i, funcIndex(L), numLayers, 0);
                emptyOutput(L).a = handler(L).layer.bn;
                L = L+1;
            end

            %% Initialize constructor
            obj.emptyOutput = emptyOutput;
            obj.numLayers = numLayers+numConvLayers;
            obj.handler = handler;
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
            
            output = obj.emptyOutput;
            output(1).a = obj.handler(1).layer.forward_pass(data);
            for i=2:obj.numLayers
                output(i).a = obj.handler(i).layer.forward_pass(output(i-1).a);
            end

            [~, ix] = max(output(end).a);
            [~, iy] = max(label);
            accuracy = mean(ix == iy);
        end

        %% Gradient Calculation
        function calc_gradient(obj, tdata, tlabel, rate, epoch)
            batch_size = single(size(tlabel, 2));
            lrate = rate / (batch_size);
            output = obj.feed_forward(tdata, tlabel);
            dout = (output(end).a - tlabel);

            for i=obj.numLayers:-1:1
                if i ~= 1
    				dout = obj.handler(i).layer.backward_pass(dout, output(i-1).a, output(i).a, batch_size, epoch, lrate);
                else
    				obj.handler(i).layer.backward_pass(dout, tdata, output(i).a, batch_size, epoch, lrate);
                end
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

                if epoch == 1, fprintf("First run: %fs\n", time); end
                if time > config.maxTime, break; end
            end
        end
    end
end