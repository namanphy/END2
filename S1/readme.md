
# Session 1 - QnA

1. What is a neural network neuron?
1. **A neuron is the smallest and individual unit in a neural network that 
holds a particular value and is responsible to store it and pass wherever required.**
The computation unit corresponding to neuron consists of two elements, a weight, 
and an activation function.


2. What is the use of the learning rate?
2. Learning rate allows to control and limit the impact of gradients and errors on the
existing weights. If the learning rate is high, the weights will be changing by larger amount,
whereas if the LR is low the weights might be decreasing slowly then they should.
**One of the primary focus in training of neural network is to obtain a optimal Learning
Rate for the problem.**

3. How are weights initialized?
3. Weights are initialized randomly. This is done so as to reduce the entropy while calculating 
best possible values for the weights. Essentially our goal is to decrease the distance between the 
weight's best fit value and our initialised value. And **by initialising them randomly we reduce our 
bias and on average the total distance travelled is less than if starting from the same points.**

4. What is "loss" in a neural network?
4. The loss is a measure to calculate and quantify function quantifies how “good” or “bad” a 
given model is in classifying the input data. **In a simple setup networks, the loss can be calculated 
as the difference between the actual output and the predicted output.** Different loss functions 
return different errors for the same prediction, and have a considerable effect on the performance 
of the model. 

5. What is the "chain rule" in gradient flow?
5. The chain rule breaks complicated and nested mathematical expressions into subexpressions 
whose derivatives are easier to compute. We update any weight after calculating the derivative of 
the error wrt old weight. But **when more then one layer is involved to give output for a single neuron,
the calculation of derivative wrt each associated weights is done using chain rule.**