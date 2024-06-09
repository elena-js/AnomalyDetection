# PROCESSING AND VISUALIZATION OF DATA ----------------------------------------------------------------------

# Libraries
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('\n*DATA PREPROCESS*')

# Specify the csv files to analyze
# New folder
file_hr_1 = 'dataset-fitness/FitabaseData3.12.16-4.11.16-new/heartrate_seconds_merged.csv'
file_min_int_1 = 'dataset-fitness/FitabaseData3.12.16-4.11.16-new/minuteIntensitiesNarrow_merged.csv'
file_min_steps_1 = 'dataset-fitness/FitabaseData3.12.16-4.11.16-new/minuteStepsNarrow_merged.csv'
# Original folder
file_hr_2 = 'dataset-fitness/FitabaseData4.12.16-5.12.16/heartrate_seconds_merged.csv'
file_min_int_2 = 'dataset-fitness/FitabaseData4.12.16-5.12.16/minuteIntensitiesNarrow_merged.csv'
file_min_steps_2 = 'dataset-fitness/FitabaseData4.12.16-5.12.16/minuteStepsNarrow_merged.csv'

# Read the CSV files and create DataFrames
data_hr_1 = pd.read_csv(file_hr_1)
data_int_1 = pd.read_csv(file_min_int_1)
data_steps_1 = pd.read_csv(file_min_steps_1)

data_hr_2 = pd.read_csv(file_hr_2)
data_int_2 = pd.read_csv(file_min_int_2)
data_steps_2 = pd.read_csv(file_min_steps_2)

# Concat the 2 datasets
data_hr = pd.concat([data_hr_1, data_hr_2], ignore_index=True)
data_int = pd.concat([data_int_1, data_int_2], ignore_index=True)
data_steps = pd.concat([data_steps_1, data_steps_2], ignore_index=True)

def adjust_data(data_hr, data_int, data_steps):
    # Rename the column 'Value' to 'HeartRate'
    data_hr = data_hr.rename(columns={'Value': 'HeartRate'})

    # Adjust the format of the time column to the type datetime
    data_hr['Time'] = pd.to_datetime(data_hr['Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    data_int['ActivityMinute'] = pd.to_datetime(data_int['ActivityMinute'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    data_steps['ActivityMinute'] = pd.to_datetime(data_steps['ActivityMinute'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    # Round the ActivityMinute column to have the notation hh:min:00
    data_int['ActivityMinute'] = data_int['ActivityMinute'].dt.floor('T')
    data_steps['ActivityMinute'] = data_steps['ActivityMinute'].dt.floor('T')

    # Create a new column in heartrate data that represents the minute of each sample
    data_hr['ActivityMinute'] = data_hr['Time'].dt.floor('T')

    # Show the data
    #print(data_hr)
    #print(data_int)
    #print(data_steps)

    # Print main info about the data
    #print(data_hr.info())
    #print(data_int.info())
    #print(data_steps.info())

    return data_hr, data_int, data_steps

data_hr, data_int, data_steps = adjust_data(data_hr, data_int, data_steps)
### In data_hr we have one value per sample (every x sec)


# USERS ANALYSIS ----------------------------------------------------------------------------

# Obtain all users
users_hr = data_hr['Id'].unique()
users_int = data_int['Id'].unique()
users_step = data_steps['Id'].unique()
common_users = list(set(users_hr) & set(users_int) & set(users_step))
#print(f'\nCommon users: {len(common_users)}')
#print(common_users)


# In hr file: 14 users
# [2022484408 2026352035 2347167796 4020332650 4388161847 4558609924 5553957443 5577150313 6117666160 6775888955 6962181067 7007744171 8792009665 8877689391]

# 2022484408: 31 days, 2026352035: 4 days, 2347167796: 18 days, 4020332650: 16 days, *4388161847*: 30 days (28),
# 4558609924: 31 days, 5553957443: 31 days, 5577150313: 28 days, 6117666160: 23 days, 6775888955: 18 days, 
# 6962181067: 31 days, 7007744171: 24 days, 8792009665: 18 days, 8877689391: 31 days
## (in other files there are more users)

# COMMON USERS (hr, int and steps): 14
# [8792009665, 5553957443, 4558609924, 2026352035, 5577150313, 4020332650, 6962181067, 7007744171, 8877689391, 6117666160, 2347167796, *4388161847*, 2022484408, 6775888955]

# In NEW files: 14 users (hr, int and steps)
# [8792009665, 5553957443, 4558609924, 2026352035, 5577150313, 4020332650, 6962181067, 7007744171, 8877689391, 6117666160, 2347167796, 2022484408, 6775888955, *6391747486*]

# 2022484408: 12 days, 2026352035: 1 days, 2347167796: 14 days, 4020332650: 12 days, 4558609924: 12 days,
# 5553957443: 12 days, 5577150313: 11 days, 6117666160: 8 days, *6391747486*: 3 days, 6775888955: 8 days, 
# 6962181067: 14 days, 7007744171: 12 days, 8792009665: 12 days, 8877689391: 12 days


### COMMON USERS OF ORIGINAL AND NEW FILES: 13 (hr, int and steps)
# 2022484408: 42 days (28), 2026352035: 5 days (0), 2347167796: 32 days (21), 4020332650: 27 days (6), 
# 4558609924: 42 days (9), 5553957443: 42 days (25), 5577150313: 39 days (27-error), 6117666160: 31 days (21), 
# 6775888955: 26 days (3), 6962181067: 44 days (17), 7007744171: 35 days (26), 8792009665: 29 days (18), 
# 8877689391: 42 days (37)


# Analyze the data of one specific user -----------------------------------------------------
user = 8877689391
print(f'\nUSER {user}')
## The rest of the users have missing values in other files (int, steps,...) - solve it


def obtain_data_user(user):
    # Obtain heart rate, intensities and steps data
    data_hr_user = data_hr[data_hr['Id'] == user]
    data_int_user = data_int[data_int['Id'] == user]
    data_steps_user = data_steps[data_steps['Id'] == user]
    
    return data_hr_user, data_int_user, data_steps_user

# Obtain a dataframe for each feature with the info of a specific user
data_hr_user, data_int_user, data_steps_user = obtain_data_user(user)


# VISUALIZE DATA ----------------------------------------------------------------------------

# Visualize ALL the data
def visualize_all_data(user, data_hr_user):
    
    plt.figure(figsize=(20, 10))
    x = data_hr_user['Time']
    y = data_hr_user['HeartRate']
    plt.scatter(x, y)

    # The regular values are between 60 and 100
    plt.axhline(y=60, color='red', linestyle='--', label='60 bpm')
    plt.axhline(y=100, color='red', linestyle='--', label='100 bpm')
    
    plt.xlabel('\nTime')
    plt.ylabel('bpm\n')
    plt.title(f'HeartRate of user {user} from {x.iloc[0].date()} to {x.iloc[-1].date()}')
    plt.ylim(30, 220)
    # Verify if the folder exists
    if not os.path.exists(f'figures/user{user}'):
        os.makedirs(f'figures/user{user}')
    # Save the figure
    plt.savefig(f'figures/user{user}/all.png', bbox_inches='tight')
    #plt.show()
    plt.clf()

visualize_all_data(user, data_hr_user)

# Comparison between days for a user (all days)
def visualize_days_data(user, data_hr_user):
    #days = [pd.to_datetime('2016-04-12'), pd.to_datetime('2016-04-13'), pd.to_datetime('2016-04-14'), pd.to_datetime('2016-04-15')]
    days = data_hr_user['Time'].dt.date.unique().tolist()
    plt.figure(figsize=(20, 10))
    # Make a plot for each day
    for day in days:
        data_hr_user_1day = data_hr_user[data_hr_user['Time'].dt.date == day]
        x = data_hr_user_1day['Time']
        y = data_hr_user_1day['HeartRate']
        plt.scatter(x, y)
        plt.axhline(y=60, color='red', linestyle='--', label='60 bpm')
        plt.axhline(y=100, color='red', linestyle='--', label='100 bpm')
        plt.xlabel('\nTime')
        plt.ylabel('bpm\n')
        plt.title(f'HeartRate of user {user} on {day}')
        plt.ylim(30, 220)
        # Verify if the folder exists
        if not os.path.exists(f'figures/user{user}'):
            os.makedirs(f'figures/user{user}')
        # Save the figure
        plt.savefig(f'figures/user{user}/{day}.png', bbox_inches='tight')
        #plt.show()
        plt.clf()

visualize_days_data(user, data_hr_user)


# OBTAIN VARIABLES FOR THE SYSTEM -----------------------------------------------------------

def obtain_variables():
    # Calculate an average value of heart rate for each minute
    min_hr = data_hr_user.groupby('ActivityMinute')['HeartRate'].mean().reset_index()
    # Create a dataframe with each day and its array of heart rates
    day_hr_df = min_hr.groupby(min_hr['ActivityMinute'].dt.date)['HeartRate'].apply(list).reset_index()
    day_hr_df.columns = ['Day', 'HeartRates']
    # List of vectors with the values of heart rate for each day
    day_hr = day_hr_df['HeartRates'].tolist()

    # Group the values of intensities and steps by days
    day_int_df = data_int_user.groupby(data_int_user['ActivityMinute'].dt.date)['Intensity'].apply(list).reset_index()
    day_int = day_int_df['Intensity'].tolist()
    day_steps_df = data_steps_user.groupby(data_steps_user['ActivityMinute'].dt.date)['Steps'].apply(list).reset_index()
    day_steps = day_steps_df['Steps'].tolist()
    
    # Obtain list with the sequence of time values for each day (for heart rate, intensities and steps)
    day_time = min_hr.groupby(min_hr['ActivityMinute'].dt.date)['ActivityMinute'].apply(list).tolist()
    day_int_time = data_int_user.groupby(data_int_user['ActivityMinute'].dt.date)['ActivityMinute'].apply(list).tolist()
    day_steps_time = data_steps_user.groupby(data_steps_user['ActivityMinute'].dt.date)['ActivityMinute'].apply(list).tolist()

    # Show the lengths of the vectors
    print("\nNumber of Days:")
    print(len(day_hr))
    print("Number of Samples for each day: HR")
    print(*(len(day) for day in day_hr))
    ### Each day has a different number of samples

    print("Number of Samples for each day: INT")
    print(*(len(day) for day in day_int))
    print("Number of Samples for each day: STEPS")
    print(*(len(day) for day in day_steps))

    return day_hr, day_time, day_int, day_int_time, day_steps, day_steps_time

day_hr, day_time, day_int, day_int_time, day_steps, day_steps_time = obtain_variables()
### Now in data_hr we have one value per minute


# DEAL WITH MISSING VALUES ------------------------------------------------------------------

# First identify the times of the missing values - fill hrs with Nan values
def fill_missing_values_with_nan(day_time, day_hr):

    # New variables for time and hr data
    day_time_with_nan = []
    day_hr_with_nan = []
    
    for day, hr in zip(day_time, day_hr):

        # Create dataframe with values of time and hr
        df = pd.DataFrame({'Time': day, 'HR': hr})

        # Create a date range from the first sample to the last one
        dates_range = pd.date_range(start=df['Time'].min(), end=df['Time'].max(), freq='T')

        # Reindex the df to include all dates in the range
        df = df.set_index('Time').reindex(dates_range).reset_index()
        df.rename(columns={'index': 'Time'}, inplace=True)
        
        # Fill missing values with NaN
        df = df.fillna(np.nan)

        # Print new dataframe
        pd.set_option('display.max_columns', None)
        #print(df)

        # Identify missing values (samples with Nan)
        nan_rows = df[df.isna().any(axis=1)]
        #print(nan_rows)

        # Add vectors to the variables (new time and hr)
        day_time_with_nan.append(df['Time'].tolist())
        day_hr_with_nan.append(df['HR'].tolist())

    return day_time_with_nan, day_hr_with_nan

day_time_with_nan, day_hr_with_nan = fill_missing_values_with_nan(day_time, day_hr)

# If there are less than x consecutive mins missing - fill with the average of previous and next values
def fill_missing_values_with_avg(max_mins_missing):
    for hrs in day_hr_with_nan:
        # Variable for counting the consecutive Nan values
        num_nans = 0
        for i, hr in enumerate(hrs):
            # We increment the counter if Nan value (also check it's not the last value)
            if np.isnan(hr) and i < len(hrs) - 1:
                num_nans += 1
            # If not Nan or last value
            else:
                # If we have no more than x consecutive nan values (no more than x min missing)
                if num_nans > 0 and num_nans <= max_mins_missing:
                    # We fill the Nan values with the average of the previous and next values
                    start_id = i - num_nans
                    end_id = i - 1
                    # If it's possible, add the previous and next values to a vector
                    prev_next_val = []
                    prev_next_val.append(hrs[start_id - 1]) if start_id - 1 >= 0 else None
                    prev_next_val.append(hrs[end_id + 1]) if end_id + 1 <= len(hrs) - 1 else None
                    # If we only have the previous or the next value
                    if len(prev_next_val) == 1:
                        fill_value = prev_next_val[0]
                    # If we have both previous and next values
                    if len(prev_next_val) == 2:
                        fill_value = (prev_next_val[0] + prev_next_val[1]) / 2
                    # Fill the missing values
                    fill_values = [fill_value] * (end_id - start_id + 1)
                    hrs[start_id:end_id+1] = fill_values
                # We restart the counter as there are no more consecutive Nan values
                num_nans = 0

    return day_hr_with_nan  
        
day_hr_with_nan = fill_missing_values_with_avg(max_mins_missing=60)

# See the results
days_with_nan = []
for day, hrs in zip(day_time_with_nan, day_hr_with_nan):
    for i, hr in enumerate(hrs):
        if np.isnan(hrs[i]):
            days_with_nan.append(day[0].strftime("%m-%d"))
days_with_nan = list(set(days_with_nan))
print(f'\nDays with many missing values: {len(days_with_nan)}')
#print(days_with_nan)

# We remove the days that still have missing values:
ids = [i for i, hr in enumerate(day_hr_with_nan) if not np.isnan(hr).any()]
day_hr_updated = [day_hr_with_nan[i] for i in ids]
day_time_updated = [day_time_with_nan[i] for i in ids]

# Show the lengths of the updated vectors
print("\nUpdated Number of Days:")
print(len(day_hr_updated))
print("Updated Number of Samples for each day:")
lens = [len(day) for day in day_hr_updated]
print(*lens)
print()
#for day in day_time_updated:
#    print([day[0], day[-1]])
### Each day has a different number of samples - from xx:xx to yy:yy

# Filter the days with more than 600 samples:
min_number_samples = 600

days_ids = [i for i, day in enumerate(day_hr_updated) if len(day) > min_number_samples]
day_time_updated = [day for id, day in enumerate(day_time_updated) if id in days_ids]
day_hr_updated = [day for id, day in enumerate(day_hr_updated) if id in days_ids]

# 1 OPTION - Maintain only hours with data for all
def filter_range_time():
    first_times = []
    last_times = []
    # Store the times of beggining and end of each day
    for day in day_time_updated:
        first_times.append(day[0].replace(year=2000, month=1, day=1))
        last_times.append(day[-1].replace(year=2000, month=1, day=1))
    # Select the range of time that is stored in all days
    first_time = max(first_times)
    last_time = min(last_times)
    print(f"First time: {first_time.strftime('%H:%M')}")
    print(f"Last time: {last_time.strftime('%H:%M')}")
    # Maintain only the values between the range of dates
    day_time_new = []
    day_hr_new = []
    for day,hr in zip(day_time_updated, day_hr_updated):
        ids = np.where(((np.array([d.replace(year=2000, month=1, day=1) for d in day])) >= first_time) & ((np.array([d.replace(year=2000, month=1, day=1) for d in day])) <= last_time))[0]
        day_time_new.append([day[id] for id in ids])
        day_hr_new.append([hr[id] for id in ids])

    return day_time_new, day_hr_new

day_time_new, day_hr_new = filter_range_time()

# 2 OPTION - Maintain only first x samples

# Function to only keep days with x samples or more (same length for all)
def equal_size_vectors(length, day_time_updated, day_hr_updated):
    # Obtain days (hrs) with more than x samples
    day_ids, day_hr_new = zip(*[(id, hr) for id, hr in enumerate(day_hr_updated) if len(hr) >= length])
    # Maintain only the first x hr values
    day_hr_new = [[hr for hr in arr[:length]] for arr in day_hr_new]

    # Filter the days (times) with more than x samples
    day_time_new = [day_time_updated[i] for i in day_ids]
    # Maintain only first x time values
    day_time_new = [[time for time in arr[:length]] for arr in day_time_new]

    return day_time_new, day_hr_new

def filter_num_samples():
    # All days will have the length of the day with min length
    lens = [len(hr) for hr in day_hr_updated]
    min_len = min(lens)
    # For all vectors take the x first values
    day_time_new, day_hr_new = equal_size_vectors(min_len, day_time_updated, day_hr_updated)
    return day_time_new, day_hr_new

# Use the second option if number of samples < 600
if len(day_hr_new[0]) < min_number_samples:
    day_time_new, day_hr_new = filter_num_samples()
    print('\nFILTERED BY NUMBER OF SAMPLES')
else:
    print('\nFILTERED BY TIME RANGE')

# Check number of days and samples
print(f"\nNew number of Days: {len(day_hr_new)}")
print(f"New number of Samples for each day: {len(day_hr_new[0])}\n")

# Round the heart rate values and convert to integer
day_hr_new = [[int(round(hr)) for hr in day] for day in day_hr_new]


# CALCULATE SOME STATISTICS -----------------------------------------------------------------

# Select the original data of the days that we are maintaining
dates = [day[0].date() for day in day_time_new]
data_ids = [id for id, day in enumerate(day_time) if day[0].date() in dates]
day_hr_statistics = [day for id, day in enumerate(day_hr) if id in data_ids]

# We calculate statistics from the original data (to use the real information)

def calculate_statistics(day_hr):
    # Max, Min, Mean, Median, Standard Deviation of heart rate
    max = []
    min = []
    mean = []
    median = []
    std_deviation = []
    # RMSSD: Sqrt of the mean of the squared differences of successive RR intervals) - supposed to be 20-89 ms
    rmssd = [] 
    # pNN50: % of consecutive RR interval differences that are greater than 50 ms - supposed to be 3-43 %
    pnn50 = []

    for hrs in day_hr:
        hrs = np.array(hrs)

        # Metrics from heart rates
        max.append(np.max(hrs).round().astype(int))
        min.append(np.min(hrs).round().astype(int))
        mean.append(np.mean(hrs))
        median.append(np.median(hrs))
        std_deviation.append(np.std(hrs))

        # Convert heart rates to RR intervals
        rr = 60 * 1000 / hrs

        # Metrics from RR intervals
        #rmssd_day = np.sqrt(np.mean(np.diff(day)**2))
        rmssd_day = np.sqrt(np.mean(np.square(np.diff(rr))))
        rmssd.append(rmssd_day)
        nn50 = np.sum(np.abs(np.diff(rr)) > 50)
        pnn50_day = 100 * nn50 / len(rr)
        pnn50.append(pnn50_day)

    return max, min, mean, median, std_deviation, rmssd, pnn50

max, min, mean, median, std_deviation, rmssd, pnn50 = calculate_statistics(day_hr_statistics)


# ADD OTHER FEATURES - INTENSITIES AND STEPS ------------------------------------------------

# Only maintain the info of the days we selected
day_int_new = [day for day, time in zip(day_int, day_int_time) if time[0].date() in dates]
day_int_time_new = [time for time in day_int_time if time[0].date() in dates]
day_steps_new = [day for day, time in zip(day_steps, day_steps_time) if time[0].date() in dates]
day_steps_time_new = [time for time in day_steps_time if time[0].date() in dates]

# Keep only the values corresponding to the same times as the heart rates
# Find the ids of the time samples that we have to keep
int_ids = [[time_int.index(x) for x in time if x in time_int] for time, time_int in zip(day_time_new, day_int_time_new)]
steps_ids = [[time_steps.index(x) for x in time if x in time_steps] for time, time_steps in zip(day_time_new, day_steps_time_new)]

# Only maintain the values that correspond to the ids
day_int_new = [[day for id, day in enumerate(int_day) if id in ids] for ids, int_day in zip(int_ids, day_int_new)]
day_steps_new = [[day for id, day in enumerate(steps_day) if id in ids] for ids, steps_day in zip(steps_ids, day_steps_new)]

# Check if the lengths of the different features match
if len(day_time_new) != len(day_int_new): print('\n***MISSING INTENSITY VALUES')
if len(day_time_new) != len(day_steps_new): print('\n***MISSING STEPS VALUES')

# Add a variable with the total steps taken each day
day_steps_count = [sum(day) for day in day_steps_new]


# SEE RELATION BETWEEN FEATURES -------------------------------------------------------------

# Plot HeartRate / Intensity
def relation_hr_int():
    plt.figure(figsize=(20, 10))
    # Make a plot for each day
    for id,day in enumerate(dates):
        x = day_hr_new[id]
        y = day_int_new[id]
        # Check if the lengths of the features match
        if len(x) != len(y): 
            print(f'*Lengths of heart rates and intensities do not match - graphic not created')
            return
        plt.scatter(x, y, alpha=0.5)
        plt.xlabel('\nHeartRate (bpm)')
        plt.ylabel('Intensity (0: sedentary - 3: very active)\n')
        plt.title(f'HeartRate/Intensity of user {user} on {day}')
        plt.xlim(30, 220)
        plt.ylim(-0.5, 3.5)
        plt.yticks([0, 1, 2, 3])
        plt.grid(True)
        if not os.path.exists(f'figures/user{user}'):
            os.makedirs(f'figures/user{user}')
        # Save the figure
        plt.savefig(f'figures/user{user}/hr_int_{day}.png', bbox_inches='tight')
        #plt.show()
        plt.clf()

relation_hr_int()

# Plot HeartRate and Intensity by Time
def relation_hr_int_time():
    plt.figure(figsize=(20, 10))
    # Make a plot for each day
    for id,day in enumerate(dates):
        x = day_time_new[id]
        y = day_int_new[id]
        # Check if the lengths of the features match
        if len(x) != len(y): 
            print(f'*Lengths of heart rates and intensities do not match - graphic not created')
            return
        # Change the scale to see it correctly in the image
        y = [value*50 for value in y]
        plt.scatter(x, y, label='Intensity', alpha=0.5)
        x = day_time_new[id]
        y = day_hr_new[id]
        plt.plot(x, y, label='HeartRate',color='orange', alpha=0.5)
        plt.xlabel('\nTime')
        plt.ylabel('Intensity (0: sedentary - 3: very active)\n\nHeartRate (bpm)\n')
        plt.title(f'Intensity and HR of user {user} on {day}')
        plt.ylim(-5, 220)
        plt.grid(True)
        plt.legend()
        if not os.path.exists(f'figures/user{user}'):
            os.makedirs(f'figures/user{user}')
        # Save the figure
        plt.savefig(f'figures/user{user}/hr_int_time_{day}.png', bbox_inches='tight')
        #plt.show()
        plt.clf()

relation_hr_int_time()

# Plot HeartRate / Steps
def relation_hr_steps():
    plt.figure(figsize=(20, 10))
    # Make a plot for each day
    for id,day in enumerate(dates):
        x = day_hr_new[id]
        y = day_steps_new[id]
        # Check if the lengths of the features match
        if len(x) != len(y): 
            print(f'*Lengths of heart rates and steps do not match - graphic not created')
            return
        plt.scatter(x, y, alpha=0.5)
        plt.xlabel('\nHeartRate (bpm)')
        plt.ylabel('Steps\n')
        plt.title(f'HeartRate/Steps of user {user} on {day}')
        plt.xlim(30, 220)
        plt.ylim(-5, 155)
        plt.grid(True)
        if not os.path.exists(f'figures/user{user}'):
            os.makedirs(f'figures/user{user}')
        # Save the figure
        plt.savefig(f'figures/user{user}/hr_step_{day}.png', bbox_inches='tight')
        #plt.show()
        plt.clf()

relation_hr_steps()

# Plot HeartRate and Steps by Time
def relation_hr_steps_time():
    plt.figure(figsize=(20, 10))
    # Make a plot for each day
    for id,day in enumerate(dates):
        x = day_time_new[id]
        y = day_steps_new[id]
        # Check if the lengths of the features match
        if len(x) != len(y): 
            print(f'*Lengths of heart rates and steps do not match - graphic not created')
            return
        plt.plot(x, y, label='Steps', alpha=0.5)
        x = day_time_new[id]
        y = day_hr_new[id]
        plt.plot(x, y, label='HeartRate', alpha=0.5)
        plt.xlabel('\nTime')
        plt.ylabel('Steps\n\nHeartRate (bpm)\n')
        plt.title(f'Steps and HR of user {user} on {day}')
        plt.ylim(-5, 220)
        plt.grid(True)
        plt.legend()
        if not os.path.exists(f'figures/user{user}'):
            os.makedirs(f'figures/user{user}')
        # Save the figure
        plt.savefig(f'figures/user{user}/hr_step_time_{day}.png', bbox_inches='tight')
        #plt.show()
        plt.clf()

relation_hr_steps_time()


# CREATE DATAFRAME WITH THE RELEVANT DATA ---------------------------------------------------

data = pd.DataFrame()
data['Date'] = dates
data['Time'] = day_time_new
data['HeartRate'] = day_hr_new
data['Intensity'] = day_int_new
data['Steps'] = day_steps_new
data['StepsCount'] = day_steps_count
data['Max'] = max
data['Min'] = min
data['Mean'] = mean
data['Median'] = median
data['Std_dev'] = std_deviation
data['RMSSD'] = rmssd
data['pNN50'] = pnn50

# Show all columns of the dataframe
pd.set_option('display.max_columns', None)
print()
print(data)
print()

# Store the data maintaining the formats
with open(f'data_user{user}.pkl', 'wb') as f:
    pickle.dump(data, f)

# -----------------------------------------------------------------------------------------------------------