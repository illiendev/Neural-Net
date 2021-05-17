## Possible improvements
- Keeping track of accuracy in mini batches
- One massive multiplication from input to output, no in between
- Softmax function for weight calculation. Weight is 1/(1 + exp(-w))
- Degree of certainty for the presence of a sample in a mini batch. For 10
results, we have: 0.9 -0.9^(1+x). It could be further scaled by the
variance between each sample.
- Dense matrix representation for SIMD instructions. Maybe a function map
will also be needed.

- Maybe it's worth looking into a new way to calculate derivatives by first
observing what the network does with regular backpropagation and adding a
term that handles some corrections. One example of this could be a
polynomial function for each weight change through NARMAX LSR (rateFunc)

## Biological concepts
- Concept of blinking for dropout, where a layer is calculated twice with
different dropouts and results are averaged.

- Concept of social bonding for networks, where each network teaches the
other the mistakes it made.

- Priority of information: forgetting buffer. Each training batch could be
thought of as a lesson. If the lesson is correct, it will be worth
repeating so by adding a forget buffer and keeping track of the past
gradients and the net at the start of the epoch, we could calculate the
end result through this. The window of change could be smaller too, where
the changes are enacted in like 5-6 batches. 

## New Initialization
- Initialization should be its own stage where the batch size is enormous
and multiple nets are calculated in parallel to compare performance. For
that, we'll need to assess whether or not any conclusions can be derived
from Accuracy/Time metrics. (Maybe MATLAB parallel toolbox will help?)

- Normalization introduces a known domain for the previous function. It is
then possible to pre-calculate all of the operations and perform
interpolation on the results from a lookup-table.

## To do
- Investigate NAN Issue, implement convolution (and regularization if
necessary) until network has 99.6- accuracy. Then document the network,
post to GIT and implement MATLAB app.

- THEN read up about proximal policy optimization and newer technologies,
test algorithm on CIFAR-10 and new MNIST problem and start working on
data processing for the BMI data.

- Matlab APP needs to have configurable real time hyper-parameters, and net
surf function that updates every X epoch interval. 