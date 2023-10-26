import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r'dataset\NGSIM_5++.csv')

# Select the desired columns
df = df[['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X',
         'Local_Y', 'v_Vel', 'v_Acc', 'Lane_ID']]

# Handling null values
df.dropna(inplace=True)

# Sorting values by the Global_Time in ascending order
df_sorted = df.sort_values(by='Global_Time')

# Initialize lists to store lateral velocity and acceleration values
lateral_velocity = []
lateral_acceleration = []

# Loop through the DataFrame to calculate lateral velocity and acceleration
for vehicle_id, group in df_sorted.groupby('Vehicle_ID'):
    delta_y = group['Local_Y'].diff()  # Calculate change in Local_Y
    delta_t = group['Global_Time'].diff()  # Calculate time interval

    # Calculate lateral velocity (V_y)
    vy = delta_y / delta_t

    # Calculate lateral acceleration (A_y)
    ay = vy.diff() / delta_t

    # Append the lateral velocity and acceleration values to the respective lists
    lateral_velocity.extend(vy.tolist())
    lateral_acceleration.extend(ay.tolist())

# Add lateral velocity and acceleration columns to the DataFrame
df_sorted['Lateral_Velocity'] = lateral_velocity
df_sorted['Lateral_Acceleration'] = lateral_acceleration

# Initialize a list to store yaw angle values
yaw_angles = []

# Loop through the DataFrame to calculate yaw angles
for vehicle_id, group in df_sorted.groupby('Vehicle_ID'):
    delta_x = group['Local_X'].diff()  # Change in Local_X
    delta_y = group['Local_Y'].diff()  # Change in Local_Y

    # Calculate yaw angles using arctan
    yaw_angle = np.arctan2(delta_y, delta_x)

    # Append the yaw angle values to the list
    yaw_angles.extend(yaw_angle.tolist())

# Add yaw angle column to the DataFrame
df_sorted['Yaw_Angle'] = yaw_angles

# Display or save the updated DataFrame as needed
print(df_sorted[['Vehicle_ID', 'Frame_ID', 'Local_Y', 'Lateral_Velocity', 'Lateral_Acceleration','Yaw_Angle']])


# Find rows where Lane_ID changes compared to the previous row
df_sorted['Lane_Change'] = df_sorted['Lane_ID'] != df_sorted['Lane_ID'].shift(1)

# Define a function to classify the type of movement
def classify_movement(row):
    if row['Lane_Change']:
        if row['Lane_ID'] > df_sorted['Lane_ID'].shift(1)[row.name]:
            return 'Right'
        elif row['Lane_ID'] < df_sorted['Lane_ID'].shift(1)[row.name]:
            return 'Left'
        else:
            return 'Lane Keeping'
    else:
        return 'Lane Keeping'

# Apply the classification function
df_sorted['Movement_Type'] = df_sorted.apply(classify_movement, axis=1)

# Define a function to filter rows within a 10-second window
def filter_by_time(row):
    time_window = 10 * 1000  # 10 seconds in milliseconds
    time_reference = row['Global_Time']
    return df_sorted[(df_sorted['Global_Time'] >= time_reference - time_window) &
                     (df_sorted['Global_Time'] <= time_reference + time_window)]

# Apply the filter to rows where Lane_Change is True
df_sorted['Left_Movement'] = df_sorted[df_sorted['Lane_Change']].apply(filter_by_time, axis=1)
df_sorted['Right_Movement'] = df_sorted[df_sorted['Lane_Change']].apply(filter_by_time, axis=1)

# Drop unnecessary columns
df_sorted.drop(['Lane_Change'], axis=1, inplace=True)

# Display the result
processed_df = df_sorted[['Vehicle_ID', 'Global_Time', 'Lane_ID', 'Movement_Type'
                          ,'Frame_ID', 'Total_Frames', 'Local_X','Local_Y', 'v_Vel',
                            'v_Acc','Lateral_Velocity','Lateral_Acceleration','Yaw_Angle']]
                

# Save the result to a CSV file
processed_df.to_csv('generated_dataset\lane_change.csv', index=False)


# Filter data based on Movement_Type
filtered_df_kept = processed_df[processed_df['Movement_Type'] == 'Lane Keeping']
filtered_df_left = processed_df[processed_df['Movement_Type'] == 'Left']
filtered_df_right = processed_df[processed_df['Movement_Type'] == 'Right']

filtered_df_left.to_csv('generated_dataset\left.csv', index=False)
filtered_df_right.to_csv('generated_dataset\\right.csv', index=False)
filtered_df_kept.to_csv('generated_dataset\kept.csv', index=False)


# Replace these file paths with the paths to your three CSV files
file_paths = ['generated_dataset\left.csv', 'generated_dataset\\right.csv', 'generated_dataset\kept.csv']

# Initialize an empty list to store DataFrames
dfs = []

# Read 2500 rows from each CSV file and append to the list
for file_path in file_paths:
    df = pd.read_csv(file_path, nrows=2500)
    dfs.append(df)

# Concatenate the DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Shuffle the rows of the combined DataFrame
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

combined_df.to_csv('generated_dataset\dataset_for_model.csv',index=False)


