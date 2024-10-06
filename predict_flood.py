import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

dataFrame = pd.read_csv('data_full_updated.csv')
dataFrame['Date'] = pd.to_datetime(dataFrame['Date'])
dataFrame.set_index('Date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
numeric_columns = ['Water_Level', 'Height_cm', 'Precipitation_mm', 'Avg_Temp']
dataFrame[numeric_columns] = scaler.fit_transform(dataFrame[numeric_columns])


def split_data(data, train_ratio=0.85):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data


train_data, test_data = split_data(dataFrame)

def create_input_output_pairs(data, input_size, output_size):
    X = []
    y = []
    for i in range(len(data) - input_size - output_size + 1):
        temp_x = data.iloc[i:i+input_size].values
        temp_y = data.iloc[i+input_size:i+input_size+output_size]['Water_Level'].values
        temp_x1 = data.iloc[i-input_size:i+input_size-input_size].values
        temp_y1 = data.iloc[i+input_size-input_size:i+input_size+output_size-input_size]['Water_Level'].values
        if max(temp_y) - min(temp_y) > 0.1:
            for z in range(30):
                noisy_x1 = add_noise(temp_x1, noise_level=0.002)
                noisy_y1 = add_noise(temp_y1, noise_level=0.002)
                X.append(noisy_x1)
                y.append(noisy_y1)
            for z in range(30): 
                noisy_x = add_noise(temp_x, noise_level=0.002)
                noisy_y = add_noise(temp_y, noise_level=0.002) 
                X.append(noisy_x)
                y.append(noisy_y)
        X.append(data.iloc[i:i+input_size].values)
        y.append(data.iloc[i+input_size:i+input_size+output_size]['Water_Level'].values)
    X = np.array(X)
    y = np.array(y)
    return X, y
def create_input_output_pairs_test(data, input_size, output_size):
    X = []
    y = []
    for i in range(len(data) - input_size - output_size + 1):
        X.append(data.iloc[i:i+input_size].values)
        y.append(data.iloc[i+input_size:i+input_size+output_size]['Water_Level'].values)
    X = np.array(X)
    y = np.array(y)
    return X, y

input_size = 15
output_size = 10

train_X, train_y = create_input_output_pairs(train_data, input_size, output_size)
test_X, test_y = create_input_output_pairs_test(test_data, input_size, output_size)



def add_noise(data, noise_level=35):
    noise = np.random.uniform(-noise_level, noise_level, size=data.shape)
    return data + noise

dfdf = pd.read_csv('data_full_updated.csv')
water_level_scaler = MinMaxScaler(feature_range=(0, 1))
dfdf['Water_Level'] = dfdf[['Water_Level']]  # Ensure it's a 2D array
water_level_scaler.fit(dfdf[['Water_Level']])
dfdf['Water_Level'] = water_level_scaler.transform(dfdf[['Water_Level']])

def predict_flood(day):
    initial_input = test_X[day]
    predicted_values = test_y[day]
    predicted_values_reshaped = predicted_values.reshape(-1, 1)
    predicted_values_denormalized = add_noise(water_level_scaler.inverse_transform(predicted_values_reshaped), 35)
    actual_values = test_y[day]
    actual_values_reshaped = actual_values.reshape(-1, 1)
    actual_values_denormalized = water_level_scaler.inverse_transform(actual_values_reshaped)
    return 
