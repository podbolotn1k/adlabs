import pandas as pd
import numpy as np
import time
from datetime import datetime

def load_and_prepare_data(file_path):
    df = pd.read_csv(
        file_path,
        sep=';',
        decimal='.',
        na_values=['?'],
        low_memory=False
    )
    df_clean = df.dropna().copy()
    
    datetime_str = df_clean['Date'] + ' ' + df_clean['Time']
    df_clean.loc[:, 'DateTime'] = pd.to_datetime(datetime_str, format='%d/%m/%Y %H:%M:%S')

    df_clean.loc[:, 'Date'] = df_clean['DateTime'].dt.date
    df_clean.loc[:, 'Time'] = df_clean['DateTime'].dt.time

    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    df_clean.loc[:, numeric_cols] = df_clean[numeric_cols].apply(pd.to_numeric, errors='coerce')

    np_data = df_clean.to_numpy()

    return df_clean, np_data
          
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end-start          
    return wrapper

@time_it
def task1_pandas(df):
    return df[df['Global_active_power'] > 5]

@time_it
def task1_numpy(np_data, df_clean):
    return np_data[np_data[:, df_clean.columns.get_loc('Global_active_power')] > 5]

@time_it
def task2_pandas(df):
    return df[df['Voltage'] > 235]

@time_it
def task2_numpy(np_data, df_clean):
    return np_data[np_data[:, df_clean.columns.get_loc('Voltage')] > 235]

@time_it
def task3_pandas(df):
    condition = (df['Global_intensity'] >= 19) & (df['Global_intensity'] <= 20) & \
                (df['Sub_metering_2'] > df[['Sub_metering_1', 'Sub_metering_3']].max(axis=1))
    return df[condition]

@time_it
def task3_numpy(np_data, df_clean):
    intensity_idx = df_clean.columns.get_loc('Global_intensity')
    sub1_idx = df_clean.columns.get_loc('Sub_metering_1')
    sub2_idx = df_clean.columns.get_loc('Sub_metering_2')
    sub3_idx = df_clean.columns.get_loc('Sub_metering_3')
    
    condition = (np_data[:, intensity_idx] >= 19) & (np_data[:, intensity_idx] <= 20) & \
                (np_data[:, sub2_idx] > np.maximum(np_data[:, sub1_idx], np_data[:, sub3_idx]))
    return np_data[condition]

@time_it
def task4_pandas(df):
    sample = df.sample(n=500000, replace=False, random_state=42)
    mean_values = sample[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
    return mean_values

@time_it
def task4_numpy(np_data, df_clean):
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(np_data), size=500000, replace=False)
    sample = np_data[sample_indices]
    
    sub1_idx = df_clean.columns.get_loc('Sub_metering_1')
    sub2_idx = df_clean.columns.get_loc('Sub_metering_2')
    sub3_idx = df_clean.columns.get_loc('Sub_metering_3')

    mean_sub1 = np.mean(sample[:, sub1_idx])
    mean_sub2 = np.mean(sample[:, sub2_idx])
    mean_sub3 = np.mean(sample[:, sub3_idx])

    return mean_sub1, mean_sub2, mean_sub3

@time_it
def task5_pandas(df):
    after_18 = df[df['Time'] > pd.to_datetime('18:00:00').time()]
    high_power = after_18[after_18['Global_active_power'] > 6]
    
    group2_max = high_power[high_power['Sub_metering_2'] > high_power[['Sub_metering_1', 'Sub_metering_3']].max(axis=1)]
    
    split_idx = len(group2_max) // 2
    first_half = group2_max.iloc[:split_idx]
    second_half = group2_max.iloc[split_idx:]

    result = pd.concat([first_half.iloc[::3], second_half.iloc[::4]])
    return result

@time_it
def task5_numpy(np_data, df_clean):
    time_idx = df_clean.columns.get_loc('Time')
    power_idx = df_clean.columns.get_loc('Global_active_power')
    sub1_idx = df_clean.columns.get_loc('Sub_metering_1')
    sub2_idx = df_clean.columns.get_loc('Sub_metering_2')
    sub3_idx = df_clean.columns.get_loc('Sub_metering_3')
    
    time_18 = datetime.strptime('18:00:00', '%H:%M:%S').time()
    times = np_data[:, time_idx]
    time_mask = np.array([t > time_18 for t in times])
    
    power_mask = np_data[:, power_idx] > 6
    mask = time_mask & power_mask
    high_power = np_data[mask]
    
    group2_max = high_power[(high_power[:, sub2_idx] > high_power[:, sub1_idx]) & 
                            (high_power[:, sub2_idx] > high_power[:, sub3_idx])]
    
    split_idx = len(group2_max) // 2
    first_half = group2_max[:split_idx]
    second_half = group2_max[split_idx:]
    
    result = np.vstack([first_half[::3], second_half[::4]])
    return result

def main():
    file_path = r'D:\AD\lab4\household_power_consumption.txt'  

    try:
        df_clean, np_data = load_and_prepare_data(file_path)
    except Exception as e:
        print(f"Помилка завантаження даних: {e}")
        return

    tasks = [
        ('Task 1', task1_pandas, task1_numpy),
        ('Task 2', task2_pandas, task2_numpy),
        ('Task 3', task3_pandas, task3_numpy),
        ('Task 4', task4_pandas, task4_numpy),
        ('Task 5', task5_pandas, task5_numpy)
    ]
    
    for name, pandas_func, numpy_func in tasks:
        try:
            pandas_result, pandas_time = pandas_func(df_clean)
            numpy_result, numpy_time = numpy_func(np_data, df_clean)

            if name == 'Task 4':
                print(f"{name}:")
                print(f"  Середнє для Sub_metering_1: {pandas_result['Sub_metering_1']}")
                print(f"  Середнє для Sub_metering_2: {pandas_result['Sub_metering_2']}")
                print(f"  Середнє для Sub_metering_3: {pandas_result['Sub_metering_3']}")
                print(f"  Час виконання (Pandas): {pandas_time:.4f} сек")
                print(f"  Час виконання (NumPy): {numpy_time:.4f} сек")
            else:
                print(f"{name}:")
                print(f"  Час виконання (Pandas): {pandas_time:.4f} сек")
                print(f"  Час виконання (NumPy): {numpy_time:.4f} сек")

        except Exception as e:
            print(f"{name}: Помилка - {e}")

if __name__ == "__main__":
    main()
