- Layer should be the same class for both conv types
Which means dout should output the correct shape
Also enables convolution layers to be anywhere

- Training error should be calculated through EMA of feed forward

- Represent image as an equation since that'll allow different sizes

- Should validate function exist?
(Takes entire dataset)

- Full convolution for backward pass (zero padding)
Todo: Figure out how to get the 0 value (mirroring for now)
        Index can be full conv, then any row without 0 is valid
        Input dim, big index build function
- Layer should be the same class for both conv types
Which means dout should output the correct shape
Also enables convolution layers to be anywhere

- Store m and v as a ratio

- Synthetic gradients

- Fleshed out variable initialization routine, where the best combination is tested. 
For conv layers, the shared parameters significantly reduce the performance cost it takes to do this:

1. Determine smallest change possible (Also determine maximum value for weights and biases)
2. From this smallest change possible, attempt gradient descent through different possibilities, like for a pooling layer:
[+1 -1]
[-1 +1]

Assuming 2 variables [1 0], 2^N would determine how many possibilities



- Primes * binary = unique results

- Equation to describe image. 
Example: If the image is described as x^2 + 2x + 1, then conv with a filter also specified by 2x + 3 can result in a equation defined convolution

- Add non-linearity through rooting the square, effectively reducing two possible results in one.
Example: x^2 = 4 has two results. x = sqrt(4) = 2 or -2
but by calculating it as sqrt(abs(4)), N roots get reduced to 1

If A1*B1 = X, abs(X/B1) = sqrt(X) adds a non-linearity

Relu = 1, -1 = 1, 0
Rooted = 2, -2 = 2, 2 (relevant because of weight matrix transform to fit convolved input. 
If A (4x1) and B(4x1) originally generates 16 linear weights, then these results could be used to add non linearity

One option is to calculate ReLU output through default method then LSR the weight matrix (assuming a boolean 4x4 matrix)

So, If a 4x4 input convolved with a 2x2 kernel generates a 3x3 then a 2x2

then (4x4) indexed to 4x9 * 9x1 = 4x1

Reorder equation to sort by input and check for patterns

- Convolution should follow logical indices up to down then left to right

- Add noise to avoid var 0 (inversely proportional to pixel value)

- Add proper epsilon

- Training error should be calculated through EMA of feed forward

- Should validate function exist?


Convolutional Layer
        level, numLayers, numKernels, numWeights;
        w, bn, wn;
        avg, var, zhat, cache;
        convIndex, mapSize, layerSize;
        function obj = convolutional_layer(config, image, level, actType, numLayers)      
        function output = forward_pass(obj, data)     
        function da = backward_pass(obj, da, past_data, batch_size, epoch, lrate)
		function [index, mapSize] = build_index(image, kernel, stride)
		
Standard Layer
        level, numLayers;
        w, bn, wn;
        avg, var, zhat;
        function obj = standard_layer(config, level, actType, numLayers)      
        function output = forward_pass(obj, data)
        function da = backward_pass(obj, da, data, past_data, batch_size, epoch, lrate)

%% Ideas
% imresize at output
% imresize input to have stride equal [1.5 1.5] (resize by 2 and stride 3)
% gradient image, concatenating arrays w bn wn to create params
% refer to weights as function handles to index 
% ** bad because index access
% is slower than direct access
%
% verify that networks are differnet
% auto diff (gradient function?) 
% to reduce calculation size, steps could be repeated for random variables
% if 4 variables share the same step, in a 64 layer it'd be an enormous boost
% dot prod
%
% conv concatenated 2d images with zero padding (1 giant image with kernel size padding)
%
% Neural network to approximate convolutions (train to convolve)
% Any form of convolution will work, especially if multiple layers are required. One possibility is that the first layer is the
% full convolution and then output layer fitted to desired conv
% The layer is first trained to perform the convolution, then it's inserted into the network.
% Remains to be seen if the gradient needs to be transformed or not (could train network to transform)
% 
% Another thing is to approximate the result of multiple convolutions using a single layer
%
% This could possibly be extended to even perform the fft->ifft process
%
% Neural network to initialize network
% For it to be able to select the best initialization, it effectively needs to run the classification algorithm to know 
% which one works best. If no early indicators exist or they do not correlate with good late performance then this is useless
% Either the first run will be the measure of success or X number of runs will. Batched run could use the same code as
% parallelizing multiple networks, and an evolutionary algorithm could prove useful. X number of networks compete for the best
% performance
%
% (cost function is to improve classification)
%
% Network needs to be simplified and consolidated after getting conv to work to make this easier
%
% - Layer should be the same class for both conv types
% Which means dout should output the correct shape
% Also enables convolution layers to be anywhere
%
% - Training error should be calculated through EMA of feed forward
%
% - Represent image as an equation since that'll allow different sizes
%
% - Should validate function exist?
% (Takes entire dataset)