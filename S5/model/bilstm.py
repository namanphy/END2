import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=False):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                            num_layers=n_layers, 
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text, text_length):
        embedded = self.embedding(text)  # [batch size, sent_length] -> [batch size, sent_len, emb dim]
      
        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'), batch_first=True)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions, hid dim]
        #cell = [batch size, num layers * num directions, hid dim]

        # print('Hidden shape ; ', hidden.shape)

        dense_outputs = self.fc(hidden)  # [batch size, hid dim * num directions] -> [batch size, output dim]
        
        # print('Output shape ; ', dense_outputs.shape)
        
        # Final activation function softmax
        output = F.softmax(dense_outputs[0], dim=1)
        return output

