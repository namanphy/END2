from .lstm import LSTM
from .bilstm import BiLSTM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
