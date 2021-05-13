# Training a Feedforward Neural Network in Excel

We have here a following NN consists of 1 hidden layer having 2 neurons with 2 input and
2 output neurons.

![arch](https://github.com/namanphy/END2/blob/main/S2/images/arch.png)

|||
|--------------|---------------|
|`i1` and `i2` | Input neurons |
|`h1` and `h2` | Hidden neurons |
|`a_h1` and `a_h2` | Sigmoid output of `h1` and `h2` |
|`o1` and `o2` | Output neurons |
|`a_o1` and `a_o2` | Sigmoid output of `o1` and `o2` |
|`E_total`| It is total loss of the network. We need to reduce this in order to train our NN.|

- `w1` to `w8` are weights associated with the Network. Initialised randomly.
- `E1` and `E2` are two losses associated with `a_o1` and `a_o2` respectively.


# Equations
 
 
![meta](https://github.com/namanphy/END2/blob/main/S2/images/meta.png)

1. After a forward pass and one backward pass comes and all the weights are updated at once simultaneously
with respect to original weights in last forward pass.

2. In the backward pass, partial differential equations are evaluated for each weight wrt `E_total`.

3. Finally, every weight is updated by subtracting the weights from their derivatives limited by a factor called
**Learning rate**. 

i.e. `W_new = W_old - LR * ∂(E_total)/∂(W_old)`


# Training
Following image is from the excel file and clearly shows the reducing loss from training.

![training](https://github.com/namanphy/END2/blob/main/S2/images/training.png)

We can see the total loss reducing from the below graphs.

## Effect of LR on training

We can see the effect of LR on training below.

From below graphs, it is clear that in our example as we are increasing the LR, the total error is 
converging more rapidly. This is because here the gradient values (as seen above) are much less then 
1, so multiplying them with a larger number significantly reduces the error.

|||
|---|---|
| ![](https://github.com/namanphy/END2/blob/main/S2/images/lr0.1.png)| ![](https://github.com/namanphy/END2/blob/main/S2/images/lr0.2.png) |
| ![](https://github.com/namanphy/END2/blob/main/S2/images/lr0.5.png)| ![](https://github.com/namanphy/END2/blob/main/S2/images/lr0.8.png) |
| ![](https://github.com/namanphy/END2/blob/main/S2/images/lr1.0.png) | ![](https://github.com/namanphy/END2/blob/main/S2/images/lr2.0.png) |


-----------
