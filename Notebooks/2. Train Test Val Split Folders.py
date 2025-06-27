import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Set input CSV file path
input_csv = r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 3\SMS_Spam.csv"

# Define base output directory and subdirectories for splits
output_base_dir = r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 3\dataset_splits"
train_dir = os.path.join(output_base_dir, 'train')
val_dir = os.path.join(output_base_dir, 'val')
test_dir = os.path.join(output_base_dir, 'test')

# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv)

# Perform an 80/20 split for train+validation and test (using stratification)
train_val_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],  # Adjust the column name if different
    random_state=69420
)

# Further split the train_val set into training and validation sets (80/20 split)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,        # 0.25 x 0.8 = 0.20 of total data for validation
    stratify=train_val_df['label'],
    random_state=69420
)

# Define output CSV file paths
train_csv_path = os.path.join(train_dir, 'train.csv')
val_csv_path = os.path.join(val_dir, 'val.csv')
test_csv_path = os.path.join(test_dir, 'test.csv')

# Save the splits to their respective folders
train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
