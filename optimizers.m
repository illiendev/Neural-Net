% Possible alternative to bias fix. Instead of having to fix the
% bias because it is initialized to zero, have both m and v be
% gradually initialized from raw SGD to momentum based.
% obj.b1 = config.b1 - config.b1 * exp(-epochs/10);
% obj.b2 = config.b2 - config.b2 * exp(-epochs/10);

classdef optimizers < handle
    properties
        b1, b2, bias_fix;
        m_dw, m_dbn, m_dwn;
        v_dw, v_dbn, v_dwn;
        optimizer, epsilon = eps('single');
        aFunc, daFunc;
    end

    methods
        function [obj] = optimizers(config, actType)
            epochs = 1:config.numEpochs;

            switch actType
                case 0
                    aFunc = @(z) z; % Linear
                    daFunc = @(a) 1; % Linear
                case 1
                    aFunc = @(z) max(0, z); % Relu
                    daFunc = @(a) (a>0); % Relu
                case 2
                    aFunc = @(z) 1./(1 + exp(-z)); % Sigmoid
                    daFunc = @(a) a .* (1 - a); % Sigmoid
                case 3
                    aFunc = @(z) z./(1 + exp(-z)); % Swish
                    daFunc = @(a) (1.421 .* a + 0.4401) ./ (a + 0.8752); % Swish approximation (assumes a domain = [-1 1])
                case 4
                    aFunc = @(z) exp(z)./sum(exp(z)); % Softmax
                    daFunc = @(a) a .* (1 - a); % Softmax
            end

            obj.aFunc = aFunc;
            obj.daFunc = daFunc;
            obj.optimizer = config.optimizer;
            obj.b1 = single(config.beta(1));
            obj.b2 = single(config.beta(2));
            obj.bias_fix = (sqrt(1-obj.b2.^epochs) ./ (1 - obj.b1.^epochs));
            obj.m_dw = 0; obj.m_dbn = 0; obj.m_dwn = 0;
            obj.v_dw = 0; obj.v_dbn = 0; obj.v_dwn = 0;
        end

        function sgd(obj, dw, dwn, dbn, lrate)
            obj.w = obj.w - lrate * dw;
            obj.wn = obj.wn - lrate * dwn;
            obj.bn = obj.bn - lrate * dbn;
        end

        function sgd_mom(obj, dw, dwn, dbn, lrate)
            obj.m_dbn = (obj.b1 * obj.m_dbn + lrate * (1-obj.b1) * dbn);
            obj.m_dwn = (obj.b1 * obj.m_dwn + lrate * (1-obj.b1) * dwn);
            obj.m_dw =  (obj.b1 * obj.m_dw  + lrate * (1-obj.b1) * dw);
            obj.bn = obj.bn - obj.m_dbn;
            obj.wn = obj.wn - obj.m_dwn;
            obj.w =  obj.w -  obj.m_dw;
        end

        function adam(obj, dw, dwn, dbn, lrate, epoch)
            obj.m_dbn = (obj.b1 * obj.m_dbn + lrate * (1-obj.b1) * dbn);
            obj.m_dwn = (obj.b1 * obj.m_dwn + lrate * (1-obj.b1) * dwn);
            obj.m_dw =  (obj.b1 * obj.m_dw  + lrate * (1-obj.b1) * dw);
            obj.v_dbn = (obj.b2 * obj.v_dbn + lrate * (1-obj.b2) * (dbn).^2);
            obj.v_dwn = (obj.b2 * obj.v_dwn + lrate * (1-obj.b2) * (dwn).^2);
            obj.v_dw =  (obj.b2 * obj.v_dw  + lrate * (1-obj.b2) * (dw).^2);
            obj.bn = obj.bn - obj.bias_fix(epoch) .* obj.m_dbn ./ (sqrt(obj.v_dbn) + obj.epsilon);
            obj.wn = obj.wn - obj.bias_fix(epoch) .* obj.m_dwn ./ (sqrt(obj.v_dwn) + obj.epsilon);
            obj.w =  obj.w  - obj.bias_fix(epoch) .* obj.m_dw  ./ (sqrt(obj.v_dw)  + obj.epsilon);
        end
    end
end
