# Addition in ANN with MNIST
**Adding a number to MNIST digit using a neural network**

--------

# Probelm Definition
To write a neural network that can:

1. Take 2 inputs:
    - an image from MNIST dataset, and
    - a random number between 0 and 9

2. give the sum of the digit and the number as output.

The problem is approached as a classification problem which has `[0, 18]` **19 classes** to predict.


# Data Preparation

There are two types of data at hand. One is MNIST dataset and other is a required dataset for random numbers. The MNIST dataset is processed and well labeled.

**We need to create our own custom dataset here which contains and combines the two different data structures to be able to feed to the model.**

- MNIST dataset

    The dataset of MNIST contains images for the handwritten digits from 0 to 9. **The images are of size `28x28` of single channel.** 

    Each image is accompanied by a label which is a integer from 0 to 9.

- Random Number
    
    The random number that needs to be generated for addition can range from 0 to 9 and it will be a integer.


## Data Representation

1. The image is represented as a tensor of shape `28x28` and is normalized with mean and standard deviation.

2. To reprsent the random number, the random number is converted to an embedding of a one-hot vector of shape `10`.

3. A sum for the label of each image and the random number is used a ground truth or new label for the dataset.

## Data Generation

A custom Pytorch Dataset class has been created for the purpose of generating the required dataset.

From the following pseudocode, the structure of the dataset class can be easily understood.

```
class mnist_sum_dataset(Dataset):

    def __init__(self, train=True):
        mnist_dataset   : dataset for mnnist
        random_int_list : list for random numbers
        sum_list        : list of sum of mnist label and random numbers
    
    def __getitem__(self, index):
        image        <-- mnist_dataset[index]
        number_embed <-- Embedding(random_int_list[index])
        label        <-- sum_list[index]

        return [image, number_embed], label

    def __len__(self):
        return length_of_dataset

```

If we visualise some of the final results from our custom dataset:

![data](https://github.com/namanphy/END2/blob/main/S3/images/sample_data.png)


# Model Preparation

![model](https://github.com/namanphy/END2/blob/main/S3/images/model.png)

The model has two parts and the idea behind this is:

- One is for the training and getting features out of image. It is using a convolution NN with batch normalisation and maxpooling.

- Other network takes the random number embedding and the output of the conv network and is reponsible for the actual addition of numbers.

At last **logarithmic of softmax values has been taken as final output and the likelihood for the output [0,19].**

# Training

Training happened for 20 epochs. As the size of the model is very small it.

## loss
**Negative log likelihood** has been chosen as the loss for the network. The idea behind this is that this losss in combination with softmax gives good results by **penalizing the low confidence of model in incorrect class and rearding the high confidence of model in correct class.**


Below are some training logs.