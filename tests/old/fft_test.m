clc;
clear all; close all;

image = imread('peppers.png');
image = image(130:256, 130:256, :) + uint8(randi([1 64], [127 127]));
kernel = randn(3, 3, 3);

[error1] = fftconv2(image, kernel);
[error2] = fftconvn(image, kernel);

a = randn(16, 16, 3, 100);
b = randn(16, 16, 3, 1, 7);
c = a.*b;
size(c)

d = zeros(size(c));
for i=1:7
    d(:,:,:,:,i) = a.*b(:,:,:,:,i);
end

er = sum(abs(abs(c) - abs(d)) >= 1E-7, 'all');

function [error] = fftconvn(image, kernel)
valid = convn(image, kernel, 'valid');
same = convn(image, kernel, 'same');
full = convn(image, kernel, 'full');

pad = 1;
fimage = fftn(padarray(image, [0 0 0] + pad));
fkernel = fftn(padarray(kernel, [62 62 0] + pad));

x1 = fimage .* fkernel;
x2 = ifftn(x1);

xfull = ifftshift(x2);
xsame = xfull(2:end-1, 2:end-1, 2:end-1);
xvalid = xfull(3:end-2, 3:end-2, 3:end-2);

efull = sum(abs(abs(full) - abs(xfull)) >= 1E-7, 'all');
esame = sum(abs(abs(same) - abs(xsame)) >= 1E-7, 'all');
evalid = sum(abs(abs(valid) - abs(xvalid)) >= 1E-7, 'all');

error = [efull esame evalid];
end

function [error] = fftconv2(image, kernel)
image = rgb2gray(image);
kernel = rgb2gray(kernel);

valid = conv2(image, kernel, 'valid');
same = conv2(image, kernel, 'same');
full = conv2(image, kernel, 'full');

fimage = fft2(padarray(image, [1 1]));
fkernel = fft2(padarray(kernel, [63 63]));

x1 = fimage .* fkernel;
x2 = ifft2(x1);

xfull = ifftshift(x2);
xsame = xfull(2:end-1, 2:end-1);
xvalid = xfull(3:end-2, 3:end-2);

efull = sum(abs(abs(full) - abs(xfull)) >= 1E-7, 'all');
esame = sum(abs(abs(same) - abs(xsame)) >= 1E-7, 'all');
evalid = sum(abs(abs(valid) - abs(xvalid)) >= 1E-7, 'all');

error = [efull esame evalid];
end
