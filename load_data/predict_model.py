import utils
import os
import tsfel
import pandas as pd



file_path = "D:\Clean\\"
file_list = []
for file in os.listdir(file_path):
    if file.endswith(".csv") and file != "Smartwatch_GravityDatum.csv" and file != "Smartwatch_LinearAccelerationDatum.csv":
        file_list.append(file)

df_list = []
for file in file_list:
    try:
        df = utils.read_data(file_path + file)
        filter_df = utils.filter_date(df, "6f76c9b33efc0bdb", "2021-01-06", "2021-01-07").drop(columns=['T', 'ParticipantId', 'FileCreationTime', 'DeviceId'])
        df_list.append(filter_df)
    except:
        continue

X_train_list = []
col_list = []
for df in df_list:
    try:
        cfg = tsfel.get_features_by_domain()
        X_train = tsfel.time_series_features_extractor(cfg, df)
        X_train_list.append(X_train)
        col_names = X_train.columns
        col_list.append(col_names)
    except:
        continue

prev_list = col_list[0]
for col in col_list[1:]:
    prev_list = set(prev_list).intersection(col)

extracted_X_train = []
for X_train in X_train_list:
    new_train = X_train[prev_list]
    extracted_X_train.append(new_train)

X_df = pd.concat(extracted_X_train, axis=0)
print(X_df.values)





