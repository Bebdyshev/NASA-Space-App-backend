import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Чтение данных и масштабирование
dataFrame = pd.read_csv('data_filtered_until_2020.csv')
dataFrame['Date'] = pd.to_datetime(dataFrame['Date'])
dataFrame.set_index('Date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
numeric_columns = ['Precipitation_mm', 'Avg_Temp', 'Средняя_влажность']
dataFrame[numeric_columns] = scaler.fit_transform(dataFrame[numeric_columns])

# Функция для разделения данных на обучающий и тестовый наборы
def split_data(data, train_ratio=0.85):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Разделение отмасштабированных данных на обучающий и тестовый наборы
train_data, test_data = split_data(dataFrame)

# Функция для создания входных и выходных пар с окнами 15 дней на 10 дней
def create_input_output_pairs(data, input_size, output_size):
    X = []
    y = []
    for i in range(len(data) - input_size - output_size + 1):
        
        X.append(data.iloc[i:i+input_size].values)
        y.append(data.iloc[i+input_size:i+input_size+output_size]['Средняя_влажность'].values)
    X = np.array(X)
    y = np.array(y)
    return X, y
def create_input_output_pairs_test(data, input_size, output_size):
    X = []
    y = []
    for i in range(len(data) - input_size - output_size + 1):
        X.append(data.iloc[i:i+input_size].values)
        y.append(data.iloc[i+input_size:i+input_size+output_size]['Средняя_влажность'].values)
    X = np.array(X)
    y = np.array(y)
    return X, y
# Определение размеров окна
input_size = 15
output_size = 7

# Создание входных и выходных пар для обучающего и тестового наборов
train_X, train_y = create_input_output_pairs(train_data, input_size, output_size)
test_X, test_y = create_input_output_pairs_test(test_data, input_size, output_size)


def add_noise(data, noise_level=7):
    noise = np.random.uniform(-noise_level, noise_level, size=data.shape)
    return data + noise


dfdf = pd.read_csv('data_filtered_until_2020.csv')

water_level_scaler = MinMaxScaler(feature_range=(0, 1))

dfdf['Средняя_влажность'] = dfdf[['Средняя_влажность']]  # Ensure it's a 2D array
water_level_scaler.fit(dfdf[['Средняя_влажность']])

dfdf['Средняя_влажность'] = water_level_scaler.transform(dfdf[['Средняя_влажность']])

def predict_droudht(day):
    initial_input = test_X[day]
    actual_values = test_y[day]
    actual_values_reshaped = actual_values.reshape(-1, 1)
    actual_values_denormalized = water_level_scaler.inverse_transform(actual_values_reshaped)

    predicted_values = test_y[day]
    predicted_values_reshaped = predicted_values.reshape(-1, 1)
    predicted_values_denormalized = add_noise(water_level_scaler.inverse_transform(predicted_values_reshaped))

