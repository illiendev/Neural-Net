classdef mnist
    properties
    end
    
    methods (Static)
        function [tdata, tlabel, vdata, vlabel] = load_data()
            load('data\MNIST\mnist_uint8.mat', 'tdata', 'tlabel', 'vdata', 'vlabel');
            %% why is it rotated??
            rsize = [13 13];
            tdata = reshape(tdata, 28, 28, 60000);
            vdata = reshape(vdata, 28, 28, 10000);

            tdata = imresize(tdata, rsize);
            vdata = imresize(vdata, rsize);
            tdata = reshape(tdata, prod(rsize), 60000);
            vdata = reshape(vdata, prod(rsize), 10000);

            %%
            tdata = single(tdata)/255;
            tlabel = single(tlabel);
            vdata = single(vdata)/255;
            vlabel = single(vlabel);
            
            tdata = (tdata - mean(tdata(:))) ./ std(tdata(:));
            vdata = (vdata - mean(vdata(:))) ./ std(vdata(:));
        end
        
        function data = new_sample(data)
            imageSize = size(data, 1);
            imageLength = round(sqrt(imageSize));
            data = reshape(data, imageLength, imageLength, []);
            index = 0:5000:60000;
            numSplits = length(index);
            angle = randi([-15 15], numSplits, 1, 'single');
            
            %%
            x = data(:,:,1);
            x = diff(sort(x(:)));
            pxVal = abs(min(x(x~=0)));
            data = data + 16 * pxVal * min(max(randn(size(data)), -2), 2);
            
            for i=1:numSplits-1
                p1 = index(i) + 1;
                p2 = index(i+1);
%                 data(:,:,p1:p2) = imgaussfilt(imrotate(data(:,:,p1:p2), angle(i), 'crop'), 0.3 * abs(randn()));
                data(:, :, p1:p2) = imrotate(data(:, :, p1:p2), angle(i), 'crop');
            end

            data = reshape(data, imageSize, []);
        end
    end
end