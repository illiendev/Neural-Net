
## Introduction
The neural network and all functions that relate to it's functioning are inside the neural_net class. One instance of the network represents one network. This allows you to easily create an ensemble with a vectorized approach.

## Configuration
The class constructor takes a single configuration structure that contains multiple settings.

	config.structure: A vector that represents the network architecture. Only the hidden layers are needed, the output layer is automatically determined.
	config.activation: A vector that maps what activation function goes to what layer. Refer to activation configuration.
	config.optimizer: A variable that sets what optimizer to run (0 for SGD, 1 for SGD with momentum, 2 for Adam)
	config.beta: Vector that represents the beta1 and beta2 variables respectively in momentum based optimizers
	config.logInterval: How often results are logged to the workspace (every X epochs)
	config.maxTime: How long the training should last
	config.numEpochs: How many epochs of training it will have
	
**Activation configuration**: 
0- Linear (ELU), 1- ReLU, 2- Sigmoid, 3- Swish, 4-Softmax

There are two ways of using this:
- Use two values, where the first value represents all hidden layers and second value represents output layer;
- Use the same amount of values as you have layers.


**Example**: 
- **Structure** = [200 200 100 100], **Activation** = [1 1 2 2 4];
	- This applies ReLU activation to the first two layers, sigmoid to the third/fourth layer, and softmax to the final layer
- **Structure** = [200 200 100 100 100 100 50], **Activation** = [1 4];
	- This applies ReLU to all hidden layers and Softmax to the output layer

## Variables and structures

**Layer**: structure that represents the weights and biases of the network

	layer (global):
		layer(N)		- N denotes the index of the layer, where (1) would be the input layer and (end) the output layer.
		layer(N).bn		- Bias used for batch normalization
		layer(N).wn		- Weight matrix used for batch normalization
		layer(N).w		- Regular weight matrix
	
**Empty Structures**: Structure for cached initializations to the gradient and the output
	
	emptyGradient (global):
		emptyGradient(N)				
		emptyGradient(N).bn		- Partial derivatives to bn
		emptyGradient(N).wn		- Partial derivatives to wn
		emptyGradient(N).w		- Partial derivatives to w
		emptyGradient(N).da		- Activation derivative
		emptyGradient(N).dz		- Output derivative
	
	emptyOutput (global): 
		emptyOutput(N).a		- Activated outputs

**Cache**

	cache(local):
		cache(N).avg	- Activation mean for batch normalization
		cache(N).var	- Activation variance for batch normalization
		cache(N).zhat	- Normalized activation for batch normalization

**Miscellaneous**

	tdata, tlabel 	- training data and label
	vdata, vlabel 	- validation data and label
	a 				- activation
	z 				- layer output (no distinction between normalized result for memory efficiency)
	b1/b2			- beta1/2 variable for momentum calculation
	bias_fix 		- cached version for the bias fix in adam's optimizer
	funcIndex		- Maps activation functions to its desired configuration
	train			- Global boolean that shows whether the network is currently training or validating

## Functions
- **feed_forward**: Calculates the forward pass of the network
	- Receives a set of samples
	- Calculates the network's outputs
	- Caches the intermediate normalization results for backpropagation
	- Returns the accuracy of the results (will replace validate)

- **validate**: Calculates the accuracy for a given number of samples
	- Soon to be deprecated
	
- **backprop**: Trains the network for one epoch
	- Receive a set of training data
	- Shuffle it for stochastic optimizations
	- Create an index that return starting and ending positions for a mini-batch 
	(it also ensures that all data is used, regardless of how it was split)
	- Goes through every mini-batch
		- Calculates the gradient for a mini batch
		- Applies the gradient through the selected optimizer
	- Returns the network object for that epoch
	
- **calc_gradient**: Calculates the gradient for a mini-batch
	- Receives a mini-batch of data
	- Initialized the gradient structure through the cached initialization
	- Performs a forward pass to calculate the output
	- Uses the output to calculate the partial derivatives

- **sgd, sgd_mom and adam**: Applies the gradient to the network
	- SGD uses only the current gradient
	- SGD_MOM uses an exponentially decaying moving average for the gradient
	- ADAM does the same but for a squared gradient as well