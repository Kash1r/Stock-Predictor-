import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        # Define the output layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

        # Initialize the hidden state and cell state
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        # Forward propagate the LSTM
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)

        # Pass the output of the LSTM through the output layer
        predictions = self.linear(lstm_out.view(len(input_seq), -1))

        # Return the predictions
        return predictions[-1]

# Load the historical stock price data
# This data might be obtained from a financial API or a CSV file
# For day trading, you would want to use high-frequency data, such as minute-by-minute price data
data = pd.read_csv('stock_prices.csv')

# We're only interested in the 'Close' column, so we'll extract that
data = data['Close'].values

# Convert the data to floating point values
data = data.astype('float32')

# Normalize the dataset to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# Split the data into training and testing sets
# We'll use 80% of the data for training and the remaining for testing
train_size = int(len(data) * 0.80)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]

# Convert an array of values into a dataset matrix
# This function takes a dataset and a 'look_back' parameter, which determines the number of previous time steps to use as input variables to predict the next time period
def create_dataset(dataset, look_back=60):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Reshape the train and test data
# We're using a 'look_back' of 60, which means we'll use the past 60 minutes of price data to predict the next minute
# Look_back can be modified as needed.
look_back = 60
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Convert the data to PyTorch tensors
# PyTorch models require input data in the form of tensors
trainX = torch.FloatTensor(trainX)
trainY = torch.FloatTensor(trainY)
testX = torch.FloatTensor(testX)
testY = torch.FloatTensor(testY)

# Create the LSTM model
model = LSTM()

# Define the loss function and the optimizer
# We're using Mean Squared Error (MSE) as the loss function and Adam as the optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for i in range(100):
    model.train()
    optimizer.zero_grad()

    # Initialize the hidden state and cell state to zeros
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                         torch.zeros(1, 1, model.hidden_layer_size))

    # Forward pass
    y_pred = model(trainX)

    # Compute the loss
    single_loss = loss_function(y_pred, trainY)

    # Backward pass
    single_loss.backward()

    # Update the weights
    optimizer.step()

# Switch to evaluation mode
model.eval()

# Make predictions on the training data and the test data
trainPredict = model(trainX)
testPredict = model(testX)

# Invert the predictions back to the original scale
# We had normalized our dataset to the range [0, 1], but we want to compare the predictions to the original data
trainPredict = scaler.inverse_transform(trainPredict.detach().numpy().reshape(-1, 1))
trainY = scaler.inverse_transform(trainY.detach().numpy().reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict.detach().numpy().reshape(-1, 1))
testY = scaler.inverse_transform(testY.detach().numpy().reshape(-1, 1))

# Calculate the root mean squared error (RMSE) of the predictions
# This will give us a single number that summarizes how close the predicted values are to the actual values
trainScore = np.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Shift train predictions for plotting
# We must shift the predictions so that they align on the x-axis with the original dataset
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict

# Plot the original dataset, the predictions on the training dataset, and the predictions on the test dataset
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()