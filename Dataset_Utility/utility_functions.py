# utility_functions.py
import pandas as pd

def calculate_label_rate(df):
    
    # Get the total number of samples
    total_samples = len(df)
    
    # Count the occurrences of each label in the 'label' column of df dataframe.
    label_counts = df['label'].value_counts()
    # Extract the count of positive labels (label == 1) from the label_counts series.
    positive_count = label_counts.get(1, 0)
    # Extract the count of negative labels (label == 0) from the label_counts series.
    negative_count = label_counts.get(0, 0)
    # Calculate the rate of positive labels to negative labels.
    label_rate = positive_count / negative_count if negative_count != 0 else 0
    
    # Print the sizes of positive and negative samples along with the label rate, formatted to two decimal places.
    print("Total Sample size is {}, Positive Sample size is {}, Negative Sample size is {}, label rate is {:.2f}".format(total_samples, positive_count, negative_count, label_rate))


def calculate_label_rate2(df,label):
    
    # Get the total number of samples
    total_samples = len(df)
    
    # Count the occurrences of each label in the 'label' column of df dataframe.
    label_counts = df[label].value_counts()
    # Extract the count of positive labels (label == 1) from the label_counts series.
    positive_count = label_counts.get(1, 0)
    # Extract the count of negative labels (label == 0) from the label_counts series.
    negative_count = total_samples - positive_count
    # Calculate the rate of positive labels to negative labels.
    label_rate = positive_count / negative_count if negative_count != 0 else 0
    
    # Print the sizes of positive and negative samples along with the label rate, formatted to two decimal places.
    print("Total Sample size is {}, Positive Sample size is {}, Negative Sample size is {}, label rate is {:.4f}".format(total_samples, positive_count, negative_count, label_rate))


def get_train_holdout_validate(df, label, n):
    ratio = n/df.shape[0]
    df_label_0 = df[df[label] == 0]
    df_label_1 = df[df[label] == 1]
    # 對 label 為 0 的數據集進行分割
    total_samples_label_0 = round(ratio * len(df_label_0))  #len(df_label_0)
    train_size_label_0 = int(0.4 * total_samples_label_0)
    holdout_size_label_0 = int(0.4 * total_samples_label_0)
    validate_size_label_0 = int(0.2 * total_samples_label_0)

    df_label_0_train = df_label_0.sample(n=train_size_label_0, random_state=42)
    df_label_0_hold = df_label_0.drop(df_label_0_train.index).sample(n=holdout_size_label_0, random_state=42)
    df_label_0_val = df_label_0.drop(df_label_0_train.index).drop(df_label_0_hold.index).sample(n=validate_size_label_0, random_state=42)

    # 對 label 為 1 的數據集進行分割
    total_samples_label_1 = round(ratio * len(df_label_1))
    train_size_label_1 = int(0.4 * total_samples_label_1)
    holdout_size_label_1 = int(0.4 * total_samples_label_1)
    validate_size_label_1 = int(0.2 * total_samples_label_1)

    df_label_1_train = df_label_1.sample(n=train_size_label_1, random_state=42)
    df_label_1_hold = df_label_1.drop(df_label_1_train.index).sample(n=holdout_size_label_1, random_state=42)
    df_label_1_val = df_label_1.drop(df_label_1_train.index).drop(df_label_1_hold.index).sample(n=validate_size_label_1, random_state=42)
    # 將 label 為 0 和 label 為 1 的 train 子數據集合併成 df_train, 將df_train的行隨機重新排序
    df_train = pd.concat([df_label_0_train, df_label_1_train])
    df_train_shuffled = df_train.sample(frac=1).reset_index(drop=True)
    # 將 label 為 0 和 label 為 1 的 holdout 子數據集合併成 df_holdout, 將df_holdout的行隨機重新排序
    df_holdout = pd.concat([df_label_0_hold, df_label_1_hold])
    df_holdout_shuffled = df_holdout.sample(frac=1).reset_index(drop=True)
    # 將 label 為 0 和 label 為 1 的 validate 子數據集合併成 df_val, 將df_val的行隨機重新排序
    df_val = pd.concat([df_label_0_val, df_label_1_val])
    df_val_shuffled = df_val.sample(frac=1).reset_index(drop=True)
    
    return df_train_shuffled, df_holdout_shuffled, df_val_shuffled




