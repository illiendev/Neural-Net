
// Setting sample size (Amount of distinct inputs, batch size)
// y returns how many times all inputs where present, z is %
F(x) = 10 - 9*0.9^N;
for i=1:1000000
    x = randi([1 10], 64, 1);
    for j=1:10
        temp(j) = any(x==j);
    end
    y(i) = sum(temp);
end
z = mean(y);
w = 100*mean(y==10);

a = 0;
for i=0:29
    a = a + 0.9^i;
end

