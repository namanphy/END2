import torch
import math
import time
import os


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_iterator, test_iterator, optimizer, criterion, clip=0, epochs=10):
    train_losses = []
    test_losses = []
    train_perplexity = []
    test_perplexity = []

    N_EPOCHS = epochs
    best_valid_loss = float('inf')
    
    CHECK_FOLDER = os.path.isdir('../weights/')
    print(CHECK_FOLDER)
    if not CHECK_FOLDER:
        os.makedirs('../weights/')
        print("created weights folder : ", '../weights/')

    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss = _train_epoch(model, train_iterator, optimizer, criterion, clip)
        valid_loss = _evaluate_epoch(model, test_iterator, criterion)

        train_losses.append(train_loss)
        train_perplexity.append(math.exp(train_loss))
        test_losses.append(valid_loss)
        test_perplexity.append(math.exp(valid_loss))
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
            path_model = os.path.join('../weights/', 'best_model_quora.pt')
            torch.save(model.state_dict(), path_model)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {math.exp(train_loss):.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {math.exp(valid_loss):.2f}% \n')
    
    print('Training Completed! Best model is saved in `weights` dir.')
    return train_losses, test_losses, train_perplexity, test_perplexity


def _train_epoch(model, iterator, optimizer, criterion, clip):
    
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        ques = batch.questions
        ans = batch.duplicate_questions
        
        optimizer.zero_grad()
        
        output = model(ques, ans)
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        ans = ans[1:].view(-1)
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def _evaluate_epoch(model, iterator, criterion):
    epoch_loss = 0
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            ques = batch.questions
            ans = batch.duplicate_questions

            output = model(ques, ans, 0) #turn off teacher forcing
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            ans = ans[1:].view(-1)
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, ans)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
