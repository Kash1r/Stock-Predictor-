# Necessary imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_size=20, hidden_layer_size=50, output_size=1):
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

        # Return all the predictions
        return predictions

# Load the historical stock price data
data = pd.read_csv(r"C:\Users\Kashir\Desktop\Side_Projects\TSLA.csv")  # Replace with the path to your file

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
def create_dataset(dataset, look_back=20):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Further reduce the 'look_back' parameter to 20
look_back = 20

# Reduce the size of the hidden layer in the LSTM model to 50
hidden_layer_size = 50

# Reshape the train and test data
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Convert the data to PyTorch tensors
trainX = torch.FloatTensor(trainX)
trainY = torch.FloatTensor(trainY)
testX = torch.FloatTensor(testX)
testY = torch.FloatTensor(testY)

# Create the LSTM model
model = LSTM(input_size=look_back, hidden_layer_size=hidden_layer_size)  # Adjust the input size and the hidden layer size of the model

# Define the loss function and the optimizer
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
    single_loss = loss_function(y_pred, trainY.view(-1, 1))  # Add view(-1, 1)

    # Backward pass
    single_loss.backward()

    # Update the weights
    optimizer.step()

# Make predictions on the training data and the test data
trainPredict = model(trainX)
testPredict = model(testX)

# Invert the predictions back to the original scale
trainPredict = scaler.inverse_transform(trainPredict.detach().numpy())
trainY = scaler.inverse_transform(trainY.view(-1, 1).detach().numpy())  # Add view(-1, 1)
testPredict = scaler.inverse_transform(testPredict.detach().numpy())
testY = scaler.inverse_transform(testY.view(-1, 1).detach().numpy())  # Add view(-1, 1)

# Create a time axis for the training and testing data
trainTime = np.arange(look_back, len(trainY) + look_back)
testTime = np.arange(look_back + len(trainY), len(data) - 1)

# Plot the actual data
plt.figure(figsize=(15,7))
plt.plot(np.append(trainY, testY), label='Actual')

# Create a time axis for the training and testing data
testTime = np.arange(len(trainY) + look_back, len(trainY) + len(testPredict) + look_back)

# Plot the predictions for the training and testing data
plt.plot(trainTime, trainPredict, label='Train Predict')
plt.plot(testTime, testPredict, label='Test Predict')

plt.title('Stock Price Prediction')
plt.xlabel('Time Step')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.show()
