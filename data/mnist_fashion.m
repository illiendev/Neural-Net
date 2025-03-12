classdef mnist_fashion
    properties
    end
    
    methods (Static)
        function [tdata, tlabel, vdata, vlabel] = load_data()
            load('data\MNIST_FASHION\mnist_fashion_uint8.mat', 'tdata', 'tlabel', 'vdata', 'vlabel');
            tdata = single(tdata)/255;
            tlabel = single(tlabel);
            vdata = single(vdata)/255;
            vlabel = single(vlabel);
            
            tdata = (tdata - mean(tdata(:))) ./ std(tdata(:));
            vdata = (vdata - mean(vdata(:))) ./ std(vdata(:));
        end
        
        function data = new_sample(data)
            data = reshape(data, 28, 28, []);
            index = 0:1000:60000;
            numSplits = length(index);
            angle = round(randn(numSplits, 1, 'single') * 5);
            
            for i=1:numSplits-1
                p1 = index(i) + 1;
                p2 = index(i+1);
                data(:,:,p1:p2) = imrotate(data(:,:,p1:p2), angle(i), 'crop');
            end
            data = reshape(data, 196, []);
        end
    end
end

