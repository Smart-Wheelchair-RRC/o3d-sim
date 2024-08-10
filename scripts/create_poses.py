import os
from tqdm import tqdm
import csv
import argparse


parser = argparse.ArgumentParser(description="Script parameters")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/scratch/kumaraditya_gupta/Datasets/wheelchair-azure-lidar-26-04-2024",
    help="Directory for dataset",
)

args = parser.parse_args()

# Define paths
dataset_path = args.dataset_dir
poses_file_path = os.path.join(dataset_path, "poses_with_id.txt")
output_path = os.path.join(dataset_path, "pose/")

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Determine the number of lines in the file to set the filename format
with open(poses_file_path, "r") as file:
    num_lines = sum(1 for line in file) - 1  # Subtract 1 for the header

# Determine the filename format based on the number of lines
filename_format = "{:d}.txt"
# if num_lines < 10:
#     filename_format = "{:01d}.txt"
# elif num_lines < 100:
#     filename_format = "{:02d}.txt"
# elif num_lines < 1000:
#     filename_format = "{:03d}.txt"
# else:
#     filename_format = "{:04d}.txt"

# Read and process the CSV file
with open(poses_file_path, "r") as file:
    csv_reader = csv.reader(file, delimiter=" ")
    next(csv_reader)  # Skip the header

    for index, row in enumerate(csv_reader):
        filename = os.path.join(output_path, row[-1] + ".txt")
        # filename = os.path.join(output_path, filename_format.format(index + 1))
        data = row[1:8]  # Exclude the 'Action' column and get the pose data
        with open(filename, "w") as output_file:
            output_file.write(" ".join(data))

print("Data saved successfully.")


# poses_file = open(poses_file_path, "r")
# poses_file.readline()

# for line in tqdm(poses_file):
#     data = line.split(" ")
#     id = data[8]
#     # id has a \n at the end
#     id = id[:-1]
#     output_file = open(output_path + id + ".txt", "a")
#     output_file.write(data[1] + " " + data[2] + " " + data[3] + " " +
#                       data[4] + " " + data[5] + " " + data[6] + " " + data[7])
#     output_file.close()

# poses_file.close()
