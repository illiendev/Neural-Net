clc;
clear all; close all;


a = zeros(4, 4);
a = complex(a);

for i=1:size(a,1)
    for j=1:size(a,2)
        a(i,j) = complex(i, j);
    end
end

b = complex(ones(4, 4), ones(4, 4));
c = a .* b;

a = 1:10;
b = 2.^(1:10)';
