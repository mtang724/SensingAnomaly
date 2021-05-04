import utils
import os
import tsfel
import pandas as pd
import utils
import numpy as np
import pickle

device_ids = ["45eb179e1d93ae0d", "576ded86316a78e6", "8043b45831a5ad45", "67c0d0b74e62bf81", "52a677f9605d9ac1", "fc41eab92fdbe033"]
email_path = "D:\Clean\Fluisense Emails.xlsx"


file_path = "D:\Clean\\"
file_list = []
for file in os.listdir(file_path):
    if file.endswith(".csv") and file != "Smartwatch_GravityDatum.csv" and file != "Smartwatch_LinearAccelerationDatum.csv":
        file_list.append(file)

ori_df_list = []
for file in file_list:
    try:
        df = utils.read_data(file_path + file)
        ori_df_list.append(df)
    except:
        continue

X_df_list = []
y_list_w = []
y_list_h = []
for device_id in device_ids:
    print(device_id)
    date_ranges, yws, yhs = utils.generate_date_and_y(email_path, device_id)
    for date_range, yw, yh in zip(date_ranges, yws, yhs):
        df_list = []
        for df in ori_df_list:
            filter_df = utils.filter_date(df, device_id, date_range[0], date_range[1]).drop(columns=['T', 'ParticipantId', 'FileCreationTime', 'DeviceId'])
            df_list.append(filter_df)

        X_train_list = []
        col_list = []
        pre_X_train = None
        for i in range(len(df_list)):
            df = df_list[i]
            try:
                cfg = tsfel.get_features_by_domain()
                X_train = tsfel.time_series_features_extractor(cfg, df)
                X_train_list.append(X_train)
                col_names = X_train.columns
                col_list.append(col_names)
                pre_X_train = X_train
            except:
                X_train_list.append(pre_X_train)
                continue
        if len(col_list) != 0:
            prev_list = col_list[0]
            for col in col_list[1:]:
                prev_list = set(prev_list).intersection(col)

            extracted_X_train = []
            for X_train in X_train_list:
                new_train = X_train[prev_list]
                extracted_X_train.append(new_train)

            X_df = pd.concat(extracted_X_train, axis=0)
            print(X_df.values)
            X_df_list.append(X_df)
            y_list_w.append(yw)
            y_list_h.append(yh)

    df_column_list = []
    for df in X_df_list:
        col_names = df.columns
        df_column_list.append(col_names)

    if len(df_column_list) != 0:
        prev_list = df_column_list[0]
        for col in df_column_list[1:]:
            prev_list = set(prev_list).intersection(col)

    X_list = []
    for X_train in X_df_list:
        new_train = X_train[prev_list]
        X_list.append(new_train.values)

# data = np.column_stack([xarray, yarray])
datafile_path_x = "datafileX.txt"
datafile_path_y_w = "datafileY_w.txt"
datafile_path_y_h = "datafileY_h.txt"

xarray = np.array(X_list)
yarray = np.array(y_list_w)
yarray_h = np.array(y_list_h)
output = open(datafile_path_x, 'wb')
pickle.dump(xarray, output)
output.close()
output = open(datafile_path_y_w, 'wb')
pickle.dump(yarray, output)
output.close()
output = open(datafile_path_y_h, 'wb')
pickle.dump(yarray_h, output)
output.close()



