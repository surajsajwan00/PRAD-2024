import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import losses
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load benign data
benign =  pd.read_csv("dataset/benign.csv")

# Split benign data into train and test
X_train, X_test0 = train_test_split(benign, test_size=0.1, random_state=42)

# Load other test datasets
X_test1 = pd.read_csv("dataset/mirai.scan.csv")
X_test2 =  pd.read_csv("dataset/mirai.ack.csv")
X_test3 =  pd.read_csv("dataset/mirai.syn.csv")
X_test4 =  pd.read_csv("dataset/mirai.udp.csv")
X_test5 =  pd.read_csv("dataset/mirai.udpplain.csv")

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test0)

# Define autoencoder model
input_dim = X_train_scaled.shape[1]
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=20, batch_size=64, shuffle=True, validation_split=0.1)

training_loss = losses.mse(X_train_scaled, autoencoder(X_train_scaled))
threshold = np.mean(training_loss)+np.std(training_loss)

def predict(x, threshold=threshold, window_size=82):
    x = scaler.transform(x)
    predictions = losses.mse(x, autoencoder(x)) > threshold
    # Majority voting over `window_size` predictions
    return np.array([np.mean(predictions[i-window_size:i]) > 0.5
                     for i in range(window_size, len(predictions)+1)])

def print_stats(data, outcome):
    print(f"Shape of data: {data.shape}")
    print(f"Detected anomalies: {np.mean(outcome)*100}%")
    print()

test_data = [X_test0, X_test1, X_test2, X_test3, X_test4, X_test5]

for i, x in enumerate(test_data):
    print(i)
    outcome = predict(x)
    print_stats(x, outcome)