
# Session 1 - QnA

### What is a neural network neuron?

Ans. **A neuron is the smallest and individual unit in a neural network that 
holds a particular value and is responsible to store it and pass wherever required.**
The computation unit corresponding to neuron consists of two elements, a weight, 
and an activation function.


### What is the use of the learning rate?

Ans. Learning rate allows to control and limit the impact of gradients and errors on the
existing weights. If the learning rate is high, the weights will be changing by larger amount,
whereas if the LR is low the weights might be decreasing slowly then they should.
**One of the primary focus in training of neural network is to obtain a optimal Learning
Rate for the problem.**

### How are weights initialized?

Ans. Weights are initialized randomly. This is done so as to reduce the entropy while calculating 
best possible values for the weights. Essentially our goal is to decrease the distance between the 
weight's best fit value and our initialised value. And **by initialising them randomly we reduce our 
bias and on average the total distance travelled is less than if starting from the same points.**

### What is "loss" in a neural network?

Ans. The loss is a measure to calculate and quantify function quantifies how “good” or “bad” a 
given model is in classifying the input data. **In a simple setup networks, the loss can be calculated 
as the difference between the actual output and the predicted output.** Different loss functions 
return different errors for the same prediction, and have a considerable effect on the performance 
of the model. 

### What is the "chain rule" in gradient flow?

Ans. The chain rule breaks complicated and nested mathematical expressions into subexpressions 
whose derivatives are easier to compute. We update any weight after calculating the derivative of 
the error wrt old weight. But **when more then one layer is involved to give output for a single neuron,
the calculation of derivative wrt each associated weights is done using chain rule.**