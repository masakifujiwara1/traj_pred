name = "zara02"

# Replace 'your_path/dataset.txt' with the path to your dataset file
dataset_path = './round_dataset/' + str(name) + '/rounded.txt'

# Read the data from the dataset file
with open(dataset_path, 'r') as file:
    data = file.readlines()

# Calculate the indices for splitting
total_data = len(data)
train_end = int(0.9 * total_data)
# val_end = int(0.9 * total_data)  # This is 60% for train + 30% for val

# Split the data without shuffling
train = data[:train_end]
val = data[train_end:]
# test = data[val_end:]

# Assuming the paths below are where you want to save the splits
train_path = './round_dataset/'+ str(name) + '/' + str(name) + '_train.txt'
val_path = './round_dataset/'+str(name)+'/'+str(name)+'_val.txt'

# Save the train, val, and test sets to new txt files
with open(train_path, 'w') as file:
    file.writelines(train)
with open(val_path, 'w') as file:
    file.writelines(val)
