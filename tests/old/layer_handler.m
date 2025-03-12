classdef layer_handler < handle
    properties
        layer, layerType;
    end
    
    methods
        function obj = layer_handler()

            if layerType == 0
                layer = standard_layer(config, i, funcIndex(L), numLayers);
            else
                layer = convolutional_layer(config, layerSize, i, funcIndex(L), numConvLayers);
            end
                
            obj.layerType = layerType;
        end
    end
end

