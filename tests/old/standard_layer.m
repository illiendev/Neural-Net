classdef standard_layer < handle & optimizers
    properties
        level, numLayers;
        w, bn, wn;
        avg, var, zhat;
    end
    
    %% Layer constructor
    methods
        function obj = standard_layer(config, level, actType, numLayers)
            L1 = config.structure(level);
            L2 = config.structure(level+1);
            obj@optimizers(config, actType);
            obj.w = randn(L2, L1, 'single') / sqrt(L1);
            obj.wn = ones(L2, 1, 'single');
            obj.bn = zeros(L2, 1, 'single');
            obj.level = single(level);
            obj.numLayers = single(numLayers);
            obj.optimizer = single(config.optimizer);
        end
        
        function output = forward_pass(obj, data)
            z = obj.w * data;
            batch_size = (single(size(z, 2)));
            obj.avg = sum(z, 2) / batch_size;
            obj.var = obj.epsilon + sum((z - obj.avg).^2, 2) / batch_size;
            obj.zhat = (z - obj.avg) ./ sqrt(obj.var);
            output = obj.aFunc(obj.zhat .* obj.wn + obj.bn);
        end
        
        function da = backward_pass(obj, da, input, output, batch_size, epoch, lrate)
            if obj.level~=obj.numLayers, da = da .* obj.daFunc(output); end
            dbn = sum(da, 2);
            dwn = dbn .* obj.wn;
            dzhat = da .* obj.wn;
            dz = (batch_size .* dzhat - sum(dzhat, 2) - obj.zhat .* sum(dzhat .* obj.zhat, 2)) ./ (batch_size * obj.var);
            dw = dz * input';
            da = obj.w' * dz;
            
            switch obj.optimizer
                case 0
                    sgd(obj, dw, dwn, dbn, lrate);
                case 1
                    sgd_mom(obj, dw, dwn, dbn, lrate);
                case 2
                    adam(obj, dw, dwn, dbn, lrate, epoch);
            end
        end
    end
end