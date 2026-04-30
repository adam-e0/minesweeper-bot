import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


class MinesweeperDataset(Dataset):
    def __init__(self, csv_file):
        # Load the CSV data
        self.data = pd.read_csv(csv_file)

        # The last two columns are 'global_density' and 'safe'
        self.grid_features = self.data.iloc[:, :-2].values
        self.global_density = self.data["global_density"].values
        self.targets = self.data["safe"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the 24 neighbors for this row
        cells = self.grid_features[idx]

        # 1. Reconstruct the 5x5 grid (25 cells)
        # The center cell (0,0) is missing in the CSV. We insert it at index 12 as a hidden cell (-1).
        cells = np.insert(cells, 12, -1)

        # 2. Shift values so they are >= 0 (from [-2 to 8] -> [0 to 10])
        # This is required for PyTorch's one_hot function
        cells_shifted = torch.tensor(cells + 2, dtype=torch.long)

        # 3. One-Hot Encode the 11 possible states (0 through 10)
        # Shape becomes (25, 11)
        grid_onehot = F.one_hot(cells_shifted, num_classes=11).float()

        # 4. Reshape to image format: (Channels, Height, Width) -> (11, 5, 5)
        grid_2d = grid_onehot.view(5, 5, 11).permute(2, 0, 1)

        # Get global density and target
        density = torch.tensor([self.global_density[idx]], dtype=torch.float32)
        target = torch.tensor([self.targets[idx]], dtype=torch.float32)

        return grid_2d, density, target


class MinesweeperNet(nn.Module):
    def __init__(self):
        super(MinesweeperNet, self).__init__()

        # --- Grid Processing ---
        # Input: 11 channels (one-hot classes), 5x5 grid
        # Conv1: 3x3 kernel, no padding -> reduces 5x5 to 3x3
        self.conv1 = nn.Conv2d(in_channels=11, out_channels=32, kernel_size=3)

        # Conv2: 3x3 kernel, no padding -> reduces 3x3 to 1x1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # --- Combined Processing ---
        # The CNN outputs 64 channels of 1x1 size.
        # We flatten this to 64, then add the 1 global_density feature = 65 inputs
        self.fc1 = nn.Linear(64 + 1, 32)
        self.fc2 = nn.Linear(32, 1)  # Output a single value (logit)

    def forward(self, grid, density):
        # 1. Process the 5x5 grid
        x = F.relu(self.conv1(grid))  # Shape: (Batch, 32, 3, 3)
        x = F.relu(self.conv2(x))  # Shape: (Batch, 64, 1, 1)

        # 2. Flatten the CNN output
        x = x.view(x.size(0), -1)  # Shape: (Batch, 64)

        # 3. Concatenate global density
        x = torch.cat((x, density), dim=1)  # Shape: (Batch, 65)

        # 4. Final Classification
        x = F.relu(self.fc1(x))
        x = self.fc2(
            x
        )  # We don't apply Sigmoid here because we will use BCEWithLogitsLoss
        return x


def train_multiple_models():
    # 1. Ask user for intervals
    user_input = input("Enter the number of intervals to divide the dataset into: ")
    try:
        num_intervals = int(user_input)
        if num_intervals <= 0:
            raise ValueError
    except ValueError:
        print("Please enter a valid positive integer.")
        return

    # 2. Setup Data and Folders
    dataset_path = "./data/minesweeper_dataset_small.csv"
    print(f"Loading full dataset from {dataset_path}...")
    full_dataset = MinesweeperDataset(dataset_path)
    total_rows = len(full_dataset)

    # Create the 'models' directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Total dataset size: {total_rows} rows. Training on {device}.\n")

    # 3. Train a model for each interval
    for i in range(1, num_intervals + 1):
        # Calculate exactly how many rows this interval uses
        # e.g., interval 1 of 4 -> 25%, interval 2 of 4 -> 50%
        current_row_count = int(total_rows * (i / num_intervals))

        print(
            f"========== Model {i}/{num_intervals} | Training on {current_row_count} rows =========="
        )

        # Create a subset using only the first `current_row_count` rows
        subset = Subset(full_dataset, range(current_row_count))

        # DataLoader handles batching and shuffling
        dataloader = DataLoader(subset, batch_size=128, shuffle=True)

        # Initialize a FRESH model so they don't share learned information
        model = MinesweeperNet().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training Loop for this specific model
        epochs = 1
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct_preds = 0
            total_samples = 0

            for grid, density, target in dataloader:
                grid, density, target = (
                    grid.to(device),
                    density.to(device),
                    target.to(device),
                )

                optimizer.zero_grad()
                predictions = model(grid, density)

                loss = criterion(predictions, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds_binary = (torch.sigmoid(predictions) >= 0.5).float()
                correct_preds += (preds_binary == target).sum().item()
                total_samples += target.size(0)

            epoch_loss = total_loss / len(dataloader)
            epoch_acc = (correct_preds / total_samples) * 100

            print(
                f"  Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%"
            )

        # 4. Save the model locally
        # .pth is the standard extension for PyTorch state dictionaries
        save_path = f"models/model-{current_row_count}.pth"

        # We only save the "state_dict" (the weights/math values), which is PyTorch best practice
        torch.save(model.state_dict(), save_path)
        print(f"-> Saved completed model to: {save_path}\n")


if __name__ == "__main__":
    train_multiple_models()
