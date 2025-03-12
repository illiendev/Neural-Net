classdef layer_handler < handle
    properties
        layer, layerType;
    end
    
    methods
        function obj = layer_handler(config, layerSize, i, funcIndex, numLayers, layerType)
            if layerType == 0
                layer = standard_layer(config, i, funcIndex, numLayers);
            else
                layer = convolutional_layer(config, layerSize, i, funcIndex, numLayers);
            end
                
            obj.layerType = layerType;
            obj.layer = layer;
        end
    end
end

