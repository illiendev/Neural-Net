## Possible improvements
- LQR simplification for convolutional layer
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

## Ideas
* Initialize kernels through gaussian random distribution, but have a variable that goes from 1 to 0 generating the opposite result
* Improve matrix performance by indexing
* Control structure that contains all of the parameters
* Check how structures handle memory storage, and kernels could be saved in 3D
* Derivatives of input remove time dependency, but add sequence dependency
genetic algorithm to generate best ensemble of networks with different sizes
different ordering for the index function in the convolution. Theres no reason it needs to be square and in series. Get an AI to determine the best order.

half precision data type

LQR Regulators (steve brunton vid)

emerson, projeto de pesquisa com uma empresa
//
1. Preprocess the convolution and shape the input to the convolution size.

Pros: 
- Drastically increases the performance (up to 5x from FFT)
- Takes advantage of big matrix multiplications on the GPU
- Makes backpropagation easier and faster

Cons: 
- Increases memory requirements to a scale of: (kernel size/image size)^2 * (image size - kernel size + 1)^2
	* Number of slices is proportional to the difference between the size of the kernel and the image
	* Another way of thinking about is the amount increased depends on how many repeated elements is in the convolution
	* Convolution stride reduces the memory usage
	* If the array is shaped in the memory contiguously, pointers could reduce the size of the array by a factor of one kernel dimension * whatever the data type reduction is
- Input is fixed to a specific size and order of the kernel (or at least the entire input array must be changed to perform the convolution)


// Matlab memory performance
Matlab handles memory addresses with linear vertical indexing, meaning each row increase takes a leap proportional to column size

If there is a big matrix multiplication, it is better to store the bigger side as columns