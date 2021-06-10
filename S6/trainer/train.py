import torch
import time
import os


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_iterator, test_iterator, optimizer, criterion, accuracy_metric, epochs=10):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    N_EPOCHS = epochs
    best_valid_loss = float('inf')
    
    CHECK_FOLDER = os.path.isdir('./weights/')
    print(CHECK_FOLDER)
    if not CHECK_FOLDER:
        os.makedirs('./weights/')
        print("created weights folder : ", './weights/')

    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss, train_acc = _train_epoch(model, train_iterator, optimizer, criterion, accuracy_metric)
        valid_loss, valid_acc = _evaluate_epoch(model, test_iterator, criterion, accuracy_metric)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(valid_loss)
        test_accuracies.append(valid_acc)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
            path_model = os.path.join('./weights/', 'best_model.pt')
            torch.save(model.state_dict(), path_model)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% \n')
    
    return train_losses, test_losses, train_accuracies, test_accuracies


def _train_epoch(model, iterator, optimizer, criterion, accuracy_metric):
    
    # initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    
    # set the model in training phase
    model.train()  
    
    for batch in iterator:

        texts, texts_lengths = batch.texts   
        
        predictions = model(texts, texts_lengths).squeeze()
        
        # print(predictions.shape, batch.labels.shape)
        loss = criterion(predictions, batch.labels)    
        optimizer.zero_grad()    
        acc = accuracy_metric(predictions, batch.labels)   
        
        loss.backward()       
        optimizer.step()      
        
        # loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def _evaluate_epoch(model, iterator, criterion, accuracy_metric):
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            texts, texts_lengths = batch.texts

            predictions = model(texts, texts_lengths).squeeze()
            
            loss = criterion(predictions, batch.labels)
            acc = accuracy_metric(predictions, batch.labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
