import torch
from torch import nn
from torchinfo import summary
import pandas as pd
from pathlib import Path
import numpy as np
torch.__version__
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, save_path="modelos/best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path) 
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def train_model(model, train_dataloader=None, val_dataloader=None, criterion=nn.MSELoss(), lr=0.1,optimizer = torch.optim.Adam, epochs=50,clip_norm=None,early_stopping=None,lr_scheduler=None):
    history = {"loss": [], "val_loss": [],"epoch":[]}
    optimizer = optimizer(model.parameters(), lr=lr)
    if lr_scheduler is not None:
        scheduler,kwargs = lr_scheduler
        scheduler = scheduler(optimizer,**kwargs)
    for epoch in range(epochs):
        model.train(True)
        running_loss = 0
        for i, data in enumerate(train_dataloader):
            inputs, targets = data[0], data[1]
            optimizer.zero_grad()
            #print(inputs.size(),targets.size())
            outputs = model(inputs)
            # Compute the loss and its gradients
            loss = criterion(outputs, targets)

            running_loss += loss.item()*inputs.size(0)
            loss.backward()
            # If clipping is set
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            # Adjust learning weights
            optimizer.step()
            
        if lr_scheduler is not None:
            scheduler.step()
            
        avg_loss = running_loss / len(train_dataloader.dataset)
        history["loss"].append(avg_loss)
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vinputs, vtargets = vdata[0], vdata[1]
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vtargets)
                running_vloss += vloss.item()*vdata[0].size(0)
        avg_vloss = running_vloss / len(val_dataloader.dataset)
        
        history["val_loss"].append(avg_vloss)
        history["epoch"].append(epoch+1)
        print('Epoch {} LOSS train {} valid {}'.format(epoch + 1, avg_loss, avg_vloss))
        if early_stopping is not None:
            if early_stopping.early_stop(avg_vloss,model): 
                print("Early stopped")
                break
    return history


def plot_history(history:dict, plot_list=[], scale="linear"):
    fig = plt.figure(figsize=(14, 7))
    plt.xlabel("Epoch")
    for plot in plot_list:
        plt.plot(history["epoch"], history[plot], label=plot)
    plt.yscale(scale)
    plt.legend(fontsize=30)
    plt.show()

def evaluate_on_test_set(model_to_evaluate, test_loader):
    model_to_evaluate.eval()
    all_predictions = []
    all_reals = []
    
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data[0], data[1]
            outputs = model_to_evaluate(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_reals.append(targets.cpu().numpy())
    
    predictions_np = np.concatenate(all_predictions)
    reals_np = np.concatenate(all_reals)
    
    test_mse = mean_squared_error(reals_np, predictions_np)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(reals_np, predictions_np)
    
    return test_mse, test_rmse, test_mae

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_output = lstm_out[:, -1, :]
        prediction = self.relu(self.fc(last_time_step_output))
        return prediction

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_time_step_output = gru_out[:, -1, :]
        prediction = self.relu(self.fc(last_time_step_output))
        return prediction