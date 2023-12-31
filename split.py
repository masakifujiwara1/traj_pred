# Replace 'your_path/dataset.txt' with the path to your dataset file
dataset_path = './round_dataset/eth/rounded.txt'

# Read the data from the dataset file
with open(dataset_path, 'r') as file:
    data = file.readlines()

# Calculate the indices for splitting
total_data = len(data)
train_end = int(0.6 * total_data)
val_end = int(0.9 * total_data)  # This is 60% for train + 30% for val

# Split the data without shuffling
train = data[:train_end]
val = data[train_end:val_end]
test = data[val_end:]

# Assuming the paths below are where you want to save the splits
train_path = './dataset_split/eth/train/train.txt'
val_path = './dataset_split/eth/val/val.txt'
test_path = './dataset_split/eth/test/test.txt'

# Save the train, val, and test sets to new txt files
with open(train_path, 'w') as file:
    file.writelines(train)
with open(val_path, 'w') as file:
    file.writelines(val)
with open(test_path, 'w') as file:
    file.writelines(test)
