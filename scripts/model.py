import torch 
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

#setting seed for reproducability - literally didn't do anything :( wasted 4 hours on this 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64):
            super().__init__()

            #lstm layer
            self.lstm = nn.LSTM(input_size * 2, hidden_size, batch_first=True) #input * 2 to account for the mask
            
            #fully connected layer
            self.fc = nn.Linear(hidden_size, 1) #output to 1 since we are only predicting one parameter

        def forward(self, features, mask):

            # should concat to (batch, seq, 2*features)? -yes
            params = torch.cat([features, mask], dim=-1)

            flow, _ = self.lstm(params)
            flow = flow[:, -1, :]

            #output returned
            return self.fc(flow)
        
def plot_training(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history['train_loss'], label='Train loss') #MSE for training set 
    plt.plot(history['val_loss'], label='Validation loss') #MSE for validation set
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training history')
    plt.margins(0) #make sure data actually stars at y-axis
    plt.legend()
    plt.tight_layout()
    plt.savefig('./LSTM Files/Images/training_loss_plot.png', dpi=300) #save just in case
    plt.show()


def evaluate(model_flow, test_loaded, device, yscale,dates,y_test):
    
    model_flow.eval() #test it
    predictions = [] #track predictions
    true = [] #organize true values

    with torch.no_grad(): #pull data from test_loaded
        for features, mask, flow in test_loaded:
            features, mask = features.to(device), mask.to(device)
            guess_flow = model_flow(features, mask) #produce predicted outputs with features and mask inputs

            #update lists
            predictions.append(guess_flow.cpu())
            true.append(flow)

    #Un-normalize the data so we can see actual predictions with context
    predicted = torch.cat(predictions).squeeze().numpy()
    observed = torch.cat(true).squeeze().numpy()
    predicted_flow = yscale.inverse_transform(predicted.reshape(-1, 1)).ravel() #inverse transform = undo previous transform 
    observed_flow = yscale.inverse_transform(observed.reshape(-1, 1)).ravel()

    plot_predictions(predicted_flow,observed_flow,dates) #produce the time-series and scatter plots for evaluation

    #standard evaluation metrics, NSE was also an option but for this model NSE = R2, so it didn't feel necessary to include
    rmse = np.sqrt(np.mean((observed_flow - predicted_flow)**2))
    mae = np.mean(np.abs(observed_flow - predicted_flow))
    r2 = 1 - (np.sum((observed_flow - predicted_flow) ** 2)) / (np.sum((observed_flow - np.mean(observed_flow)) ** 2))

    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R2:   {r2:.3f}") #extra line for visuals
    print(f"Relative Error: {(mae / y_test['Discharge_cfs'].mean()):.3f}") #MAE is correlated to mean values, so I printed the general distribution to fully understand if MAE was good/bad

def plot_predictions(predicted_flow,observed_flow,dates):
    #time series
    #dates removes the first 30 days since the model starts predicting on day 31 
    plt.figure(figsize=(12, 5))
    plt.plot(dates, observed_flow, label='Observed Flow (cfs)', color = 'purple')
    plt.plot(dates, predicted_flow, label='Predicted Flow (cfs)', color = 'limegreen')
    plt.xlabel('Date')
    plt.ylabel('Streamflow (cfs)')
    plt.title('Analysis of LSTM Observed vs Predicted Streamflow for 2016-2025')
    plt.margins(0) #make sure data actually stars at y-axis
    plt.legend()
    plt.tight_layout()
    plt.savefig('./LSTM Files/Images/predictions_timeseries.png', dpi=300) #save 
    plt.show()

    #scatter analysis, used lims to make a slope line
    plt.figure(figsize=(6, 6))
    plt.scatter(observed_flow, predicted_flow, alpha=0.5,color='b')
    lims = [min(observed_flow.min(), predicted_flow.min()), max(observed_flow.max(), predicted_flow.max())]
    plt.plot(lims, lims, 'k--')
    plt.xlabel('Observed Flow (cfs)')
    plt.ylabel('Predicted Flow (cfs)')
    plt.title('Scatter Analysis of LSTM LSTM Observed vs Predicted Streamflow(cfs)')
    plt.tight_layout()
    plt.margins(0) #make sure data actually stars at y-axis
    plt.savefig('./LSTM Files/Images/predictions_scatter.png', dpi=300) #save
    plt.show()