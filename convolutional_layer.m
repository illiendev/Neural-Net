classdef convolutional_layer < handle & optimizers
    properties
        w, bn, wn;
        convIndex, deconvIndex;
        cache;
        sz;
    end

    %% Layer constructor
    methods
        function obj = convolutional_layer(config, image, level, actType, numLayers)

            % input is cols x rows x idepth
            % output is cols x rows x kdepth
            % build moving average for score to keep track of variance

            % input size    [icols, irows, idepth]
            % kernel size   [kcols, krows, kdepth]
            % output size   [kdepth, mapCols, mapRows]
            % layer state   [level, numLayers, type]

            %% Build Index
            image = ones(image);
            kernel = config.convStructure{level};
            convStride = config.convStride{level};
            [sz.icols, sz.irows, sz.idepth] = size(image);
            [sz.kcols, sz.krows, sz.kdepth] = size(kernel);

            padding = floor([sz.kcols, sz.krows]/2);
            [convIndex, mapSize, deconvIndex] = build_index(image, kernel, convStride, padding);

            sz.kernel = sz.kcols * sz.krows * sz.idepth;
            sz.map = prod(mapSize);
            sz.mcols = mapSize(1);
            sz.mrows = mapSize(2);
            sz.level = level;
            sz.layers = numLayers;

            %% Initialize class
            obj@optimizers(config, actType);
            obj.w = randn(sz.kdepth, sz.kernel, 'single') / sqrt(sz.kernel);
            obj.wn = ones(sz.kdepth, 1, 'single');
            obj.bn = zeros(sz.kdepth, 1, 'single');

            obj.convIndex = single(convIndex);
            obj.deconvIndex = single(deconvIndex);
            obj.epsilon = single(obj.epsilon);
            obj.optimizer = single(config.optimizer);
            obj.sz = sz;
        end

        function output = forward_pass(obj, data)
            if ndims(data) == 3
                obj.cache = reshape(data(:, obj.convIndex, :), obj.sz.kernel, []);
            else
                obj.cache = reshape(data(obj.convIndex, :), obj.sz.kernel, []);
            end

            if obj.sz.level == obj.sz.layers
                output = reshape(obj.aFunc(obj.w * obj.cache + obj.bn), prod([obj.sz.kdepth, obj.sz.map]), []);
            else
                output = reshape(obj.aFunc(obj.w * obj.cache + obj.bn), obj.sz.kdepth, obj.sz.map, []);
            end
        end

        function da = backward_pass(obj, da, ~, output, batch_size, epoch, lrate)
                da = da .* obj.daFunc(output);
                da = reshape(da, obj.sz.kdepth, []);
                dbn = sum(da, 2);
                dw = da * obj.cache';

                if obj.sz.level ~= 1
                    temp1 = obj.sz.kcols * obj.sz.krows * obj.sz.kdepth;
                    temp2 = [obj.sz.idepth, obj.sz.icols*obj.sz.irows, batch_size];
                    w1 = reshape(obj.w', obj.sz.kcols * obj.sz.krows, obj.sz.idepth, obj.sz.kdepth);
                    w2b = flip(permute(w1, [1 3 2]));
                    w3 = reshape(w2b, temp1, [])';
                    da = reshape(da, obj.sz.kdepth, obj.sz.map, []);
                    da = reshape(w3 * reshape(da(:, obj.convIndex, :), temp1, []), temp2);
                end

                switch obj.optimizer
                    case 0
                        sgd(obj, dw, 0, dbn, lrate);
                    case 1
                        sgd_mom(obj, dw, 0, dbn, lrate);
                    case 2
                        adam(obj, dw, 0, dbn, lrate, epoch);
                end
        end
    end
end

function [index, mapSize, deconvIndex] = build_index(image, kernel, stride, padding)
% random stride, build multiple indexes and have a chance of using stride A
% or B (iteration 1 would be 1 and 3, iteration 2 2 and 4)
[iCols, iRows, iDepth] = size(image);
[kCols, kRows, kDepth] = size(kernel);

rowStride = stride(1);
colStride = stride(2);
iSize = iCols*iRows;
kSize = kCols*kRows;

imageIndex = padarray(reshape(1:iSize, iCols, iRows), padding, 'symmetric');
[iCols, iRows] = size(imageIndex);
numCols = (1 + iCols - kCols);
numRows = (1 + iRows - kRows);
mapSize = [numCols, numRows];

i = 1;
for row=1:numRows
    for col=1:numCols
        index(:,i) = reshape(imageIndex(col:col+kCols-1, row:row+kRows-1), kSize, 1);
        i = i+1;
    end
end

deconvIndex = index(flip(reshape(1:kSize, kRows, kCols)), :);
end