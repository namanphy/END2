# Sentiment Prediction - Encoder Decoder Architecture
### On tweets dataset using LSTM in encoder decoder fashion

--------

## Solution notebook - [EncDec_LSTM_Sentiment_Classification.ipynb](https://github.com/namanphy/END2/blob/main/S6/EncDec_LSTM_Sentiment_Classification.ipynb)

# Hyperparameters and Config
- Batch size: 128
- Learning rate: 0.0001
- Epochs: 10
- Optimizer: Adam
- Criterion: cross entropy loss
- Decoder Steps: 3

# Knowing the Data
Tweets datset has three labels 
```
[ "negative", "positive", "neutral"]
```
The dataset requires preprocessing and it is found through internet that the most relevant library to pre-process tweets data is `tweet-preprocessor` so lets download it. It helped to remove URLs, Mentions, Reserved words (RT, FAV) etc.

On exploring the dataset, it was found to be imbalanced dataset. **Huge Imbalance**

![i](https://github.com/namanphy/END2/blob/main/S6/imgs/distribution-lables.png)

I attempted to build the dataset class and to use the well known Pytorch Dataloader for consuming the data. 

Alas!! that didn't pan out. I somehow was getting very slow training time - even when the batches were sorted internally.

Need to look into it definately. For the dataset preparation and cleaning using `tweet-preprocessor` have a look at this notebook, 
**Data preparation in torchtext 0.9** - [notebook](https://github.com/namanphy/END2/blob/main/S6/tweets_dataset_torchtext_0.9.ipynb)


### Dataset - train, test split of 80-20 %.

# Model

![i](https://github.com/namanphy/END2/blob/main/S6/imgs/model.png)

The model is a encoder-decoder architecture with a fully connected layer at last.

## Encoder

This is the code for the encoder class having one LSTM and an embedding layer.

```
class lstm_encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers, dropout=0.1):
        super(lstm_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)  # [BS, S, Embed]
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=1, dropout=dropout, batch_first=True)

    def forward(self, input, hidden_state, show_states=False):
        encoder_states_collector = []
        (hidden, cell) = hidden_state

        embedded_input = self.embedding(input)  # input= [BS, S] | output= [BS, S, Embed]
        embedded_input = embedded_input.view(input.size()[1], input.size()[0], -1)  # output= [S, BS, Embed]

        # looping seq length times for a batch
        for ix in range(input.size()[1]):  
            output, (hidden, cell) = self.lstm(embedded_input[ix].unsqueeze(1),   # input= [BS, 1, Embed]
                                               (hidden, cell))                    # hidden,cell= [1, BS, Hidden]
            if show_states:
                encoder_states_collector.append({'output': output,
                                                 'hidden': hidden,
                                                 'cell': cell})
        return output, (hidden, cell), encoder_states_collector

    def init_hidden_state(self, batch_size, hidden_dim, device=device):
        zeros = torch.zeros(1, batch_size, hidden_dim).to(device) # 
        return (zeros, zeros)
```

## Decoder
This is the code for the decoder class having one LSTM layer.

Decoder Steps = 3

```
class lstm_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(lstm_decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout, batch_first=True)

    def forward(self, input, hidden_state, steps=3, show_states=False):
        decoder_states_collector = []
        (hidden, cell) = hidden_state

        for i in range(steps):
            decoder_output, (hidden, cell) = self.lstm(input,               # input= [BS, 1, Hidden]
                                                       (hidden, cell))      # hidden,cell= [1, BS, Hidden]
            input = decoder_output
            if show_states:
                decoder_states_collector.append({'output': decoder_output,
                                                 'hidden': hidden,
                                                 'cell': cell})
        return decoder_output, (hidden, cell), decoder_states_collector

    def get_decoder_input(self, batch_size, hidden_dim, device=device):
        return torch.zeros(batch_size, 1, hidden_dim).to(device)
```

# Model Diagnostic 
## visualising Encoder-Decoder hidden activations

The encoder-decoder hidden activations are recorded for a test run on a sample sentence. And here is what i have found. 

![i](https://github.com/namanphy/END2/blob/main/S6/imgs/output_text.png)

NOTE : **All the states - `output, hidden, cell` are present in notebook under `state_collector` dictionary.**

## Some Encoder Hidden States - color map

![i](https://github.com/namanphy/END2/blob/main/S6/imgs/e2.png)

![i](https://github.com/namanphy/END2/blob/main/S6/imgs/e13.png)

![i](https://github.com/namanphy/END2/blob/main/S6/imgs/e19.png)

## All three Decoder Hidden States - color map

![i](https://github.com/namanphy/END2/blob/main/S6/imgs/d1.png)

![i](https://github.com/namanphy/END2/blob/main/S6/imgs/d2.png)

![i](https://github.com/namanphy/END2/blob/main/S6/imgs/d3.png)


# Results & Experiments

**Yes!! This time an experiemnt tracking tool(Weights and Biases) has been tested out. And results are good.**

< *Will upload the dashboard link here soon.* >

The model was overfitting in all the scenarios. The accuracy and loss was measured for base line
model, the gradually augmentations were added, and model architectures/hyperparams were experimented and measured for accuracy.

### Accuracy
![](https://github.com/namanphy/END2/blob/main/S6/imgs/accuracy.png)


### ACCURACY ACHIEVED - 76% in 9 epochs
But here's the catch - During classification accuracy may not be the correct metric to evaluate model's performance. Hence, the following confusion matric is plotted to gain more insight into the model's performance. Hence

![](https://github.com/namanphy/END2/blob/main/S6/imgs/confusion_matrix.png)


# Conclusion

1. The model is not overfitting but neither giving satisfactory results.
2. Only **one context vector and no other conenction of decoder from the encoder might be resulting in information loss** and eventually not giving expected results.


# Learning
- For this model approach - one must be well informed with i/p and o/p for the LSTM. What is batch size, sequence and how are they transforming internally. Learned this.
- torchtext usage - Need to test why the torchtext-0.9 loaders not giving expected speeds.
