import pandas as pd
import datetime
from datetimerange import DateTimeRange

def read_data(path):
    df = pd.read_csv(path)
    return df


def divide_day(df, label):
    feature_list = []
    # print (pd.to_datetime(df.index))
    df['T'] = pd.to_datetime(df['T'])
    groupby_time_df = df.groupby(pd.Grouper(key='T', freq='10Min'))[label].apply(list)
    feature_list += groupby_time_df.values.tolist()
    return groupby_time_df.keys(), feature_list


def filter_date(df, deviceid, start_date, end_date):
    filtered_df = df.loc[df['DeviceId'] == deviceid]
    filtered_df = filtered_df[(filtered_df['T'] > start_date) & (filtered_df['T'] < end_date)]
    # filtered_df = filtered_df.set_index(pd.DatetimeIndex(pd.to_datetime(filtered_df['T'])))
    return filtered_df

def generate_date_and_y(email_path, label, device_id):
    email_df = pd.read_excel(email_path)
    new_header = email_df.iloc[0]  # grab the first row for the header
    email_df = email_df[1:]  # take the data less the header row
    email_df.columns = new_header  # set the header row as the df header
    user_df = email_df[email_df['Device ID'] == device_id]
    start_date = user_df["Start Date"].values[0]
    end_date = user_df["End Date"].values[0]
    print(start_date, end_date)
    date_range = []
    time_range = DateTimeRange(start_date, end_date)
    for value in time_range.range(datetime.timedelta(days=1)):
        date_range.append(value.strftime("%Y-%m-%d"))
    user_id = user_df['Password'].values[0].split('U')[0]
    path = "D:\Clean\Scale Data\Scale Data\data_{}_Scale\\".format(user_id)
    path = path + "weight.csv"
    weight_df = read_data(path)
    print(weight_df)
    return date_range


# email_path = "D:\Clean\Fluisense Emails.xlsx"
# generate_date_and_y(email_path, "xx", "45eb179e1d93ae0d")


