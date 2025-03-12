% 
%                     %%
%                     da2 = single(zeros(obj.sz.idepth, obj.sz.icols, obj.sz.irows, batch_size));
%                     for i=1:obj.sz.idepth
%                         w2 = reshape(obj.w(i,:), obj.sz.kcols, obj.sz.krows, obj.sz.idepth);
%                         da2(i, :, :, :) = convn(da2, w2, 'valid');
%                     end


%% Conv method
%             input = reshape(data, obj.sz.idepth, obj.sz.icols, obj.sz.irows, []);
%             obj.cache2 = single(zeros(obj.sz.idepth, obj.sz.kdepth, obj.sz.map, 99));
% 
%             w2 = reshape(obj.w, obj.sz.kdepth, obj.sz.kcols, obj.sz.krows, obj.sz.idepth);
%             for i=1:obj.sz.kdepth
%                 for j=1:obj.sz.idepth
%                     for k=1:size(data, 2)
%                         xw = rot90(squeeze(w2(i, :, :, j)), 2);
%                         xi = squeeze(input(j, :, :, k));
%                         temp = conv2(xi, xw, 'valid');
%                         obj.cache2(i, j, :, k) = temp(:);
%                     end
%                 end
%             end
% 
%             obj.cache2 = squeeze(sum(obj.cache2, 2));
%             obj.cache2 = obj.aFunc(obj.cache2 + obj.bn);    
% 
%             t = abscomp(output(:), obj.cache2(:));
% 
%             if (t~=0)
%             as=1;
%             end

function [index, mapSize] = build_index(image, kernel, stride, padding)
% remove stride
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
mapSize = [length(1:rowStride:numCols), length(1:colStride:numRows)];

i = 1;
for row=1:rowStride:numRows
    for col=1:colStride:numCols
        index(:,i) = reshape(imageIndex(col:col+kCols-1, row:row+kRows-1), kSize, 1);
        i = i+1;
    end
end
end
            %% Convn version
            % 			input = squeeze(reshape(data, obj.sz.idepth, obj.sz.icols, obj.sz.irows, []));
            % 			obj.cache2 = squeeze(single(zeros(obj.sz.kdepth, obj.sz.map, 100)));
            %             for i=1:obj.sz.kdepth
            %                 w2 = reshape(obj.w(i,:), obj.sz.kcols, obj.sz.krows, obj.sz.idepth);
            %                 w2 = rot90(squeeze(permute(w2, [3 1 2])), 2);
            %                 obj.cache2(i, :, :) = reshape(convn(input, w2, 'valid'), obj.sz.map, []);
            %             end

%% Pooling
% Applies average pooling by changing the convolution index to apply the
% pooling before the convolution (effectively reducing cycles, same result)

if isempty(config.convStride), poolStride = [1, 1];
elseif isempty(config.convStride{level, 2}), poolStride = [1, 1];
else, poolStride = config.convStride{level, 2};
end

if isempty(config.convStructure{level, 2})
    poolSize = 1;
    pFunc = @(x) x;
else
    poolSize = numel(config.convStructure{level, 2});
    pFunc = @(x) squeeze(mean(reshape(x, numWeights, poolSize, []), 2));
    [poolRow, poolCol] = size(config.convStructure{level, 2});
end

if poolSize ~= 1
    pool = ones(poolRow, poolCol);
    [poolIndex, mapSize] = build_index(ones(mapSize), pool, poolStride);
    convIndex = convIndex(:, poolIndex(:));
end

obj.pFunc = pFunc; % same for properties
obj.poolSize = poolSize; 


%% 
%             y = obj.w * x;
%             z = 

            % 7x9x361x100 * 7x9x5
            % da = kdepth x map x nsamples
            % w = kdepth x idepth x ksize
            % dout = (map+ksize-1) x idepth x nsamples

            % da = 7 x 225 x 100
            % w = 7 x (5*9)
            % dout = 5 x 289 x 100

%             xw = single(zeros(prod(obj.mapSize + [2 2]), obj.sz.kdepth, prod(obj.mapSize)));
%             for i=1:prod(obj.mapSize)
%                 xw(obj.convIndex(:, i), :, i) = obj.w';
%             end

%             xw = reshape(xw, prod(obj.mapSize + [2 2]), []);

%             dz = reshape(da, prod([obj.sz.kdepth obj.mapSize]), batch_size);

%             da = xw * dz;

            %             lrate = lrate / 10;
            %             if obj.level ~= 1, da = obj.w' * dz; end