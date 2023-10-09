import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import pickle


class AudioDataset(Dataset):
    def __init__(self, feat_list_conv5, feat_list_pool5, feat_list_conv6, labels):
        self.feat_list_conv5 = feat_list_conv5
        self.feat_list_pool5 = feat_list_pool5
        self.feat_list_conv6 = feat_list_conv6
        self.labels = labels
        self.total_feats = [self.feat_list_conv5, self.feat_list_pool5, self.feat_list_conv6]

    def __len__(self):
        return len(self.feat_list_pool5)

    def __getitem__(self, idx):
        return (torch.tensor(self.feat_list_conv5[idx], dtype=torch.float),
                torch.tensor(self.feat_list_pool5[idx], dtype=torch.float),
                torch.tensor(self.feat_list_conv6[idx], dtype=torch.float),
                torch.tensor(self.labels[idx], dtype=torch.long))

class SingleHeadClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SingleHeadClassifier, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.block2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.block3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.block4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.block5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.block6 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(128, num_classes)

        # Residual connection
        self.downsample1 = nn.Linear(input_dim, 2048)
        self.downsample2 = nn.Linear(2048, 2048)
        self.downsample3 = nn.Linear(2048, 1024)
        self.downsample4 = nn.Linear(1024, 512)
        self.downsample5 = nn.Linear(512, 256)

    def forward(self, x):
        identity = self.downsample1(x)
        out = self.block1(x)
        out += identity

        identity = self.downsample2(out)
        out = self.block2(out)
        out += identity

        identity = self.downsample3(out)
        out = self.block3(out)
        out += identity

        identity = self.downsample4(out)
        out = self.block4(out)
        out += identity

        identity = self.downsample5(out)
        out = self.block5(out)
        out += identity

        out = self.block6(out)
        out = self.classifier(out)

        return out
    
class MultiHeadClassifier(nn.Module):
    def __init__(self, conv5_dim, pool5_dim, conv6_dim, num_classes):
        super(MultiHeadClassifier, self).__init__()
        self.conv5_classifier = SingleHeadClassifier(conv5_dim, num_classes)
        self.pool5_classifier = SingleHeadClassifier(pool5_dim, num_classes)
        self.conv6_classifier = SingleHeadClassifier(conv6_dim, num_classes)

    def forward(self, x_conv5, x_pool5, x_conv6):
        out_conv5 = self.conv5_classifier(x_conv5)
        out_pool5 = self.pool5_classifier(x_pool5)
        out_conv6 = self.conv6_classifier(x_conv6)
        return out_conv5, out_pool5, out_conv6

def main(args):
    with open(args.list_videos, "r") as fread:
        lines = fread.readlines()[1:]

    feat_list_conv5, feat_list_pool5, feat_list_conv6, label_list = [], [], [], []
    for line in lines:
        video_id, category = line.strip().split(",")
        feat_filepath_conv5 = os.path.join(args.feat_dir, "conv5", video_id + args.feat_appendix)
        feat_filepath_pool5 = os.path.join(args.feat_dir, "pool5", video_id + args.feat_appendix)
        feat_filepath_conv6 = os.path.join(args.feat_dir, "conv6", video_id + args.feat_appendix)
        if os.path.exists(feat_filepath_pool5) and os.path.exists(feat_filepath_conv5):
            feat_list_conv5.append(np.genfromtxt(feat_filepath_conv5, delimiter=";", dtype="float"))
            feat_list_pool5.append(np.genfromtxt(feat_filepath_pool5, delimiter=";", dtype="float"))
            feat_list_conv6.append(np.genfromtxt(feat_filepath_conv6, delimiter=";", dtype="float"))
            label_list.append(int(category))

    X_train_conv5, X_val_conv5, X_train_pool5, X_val_pool5, X_train_conv6, X_val_conv6, y_train, y_val = train_test_split(
        feat_list_conv5, feat_list_pool5, feat_list_conv6, label_list, test_size=0.1, random_state=42)

    train_dataset = AudioDataset(X_train_conv5, X_train_pool5, X_train_conv6, y_train)
    val_dataset = AudioDataset(X_val_conv5, X_val_pool5, X_val_conv6, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"##########\n{device}\n##########")

    conv5_dim = len(train_dataset.feat_list_conv5[0])
    pool5_dim = len(train_dataset.feat_list_pool5[0])
    conv6_dim = len(train_dataset.feat_list_conv6[0])
    
    model = MultiHeadClassifier(conv5_dim, pool5_dim, conv6_dim, num_classes=len(set(label_list))).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

    for epoch in range(200):
        model.train()
        running_loss = 0.0
        for features_conv5, features_pool5, features_conv6, labels in train_loader:
            features_conv5,features_pool5, features_conv6, labels = features_conv5.to(device), features_pool5.to(device), features_conv6.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs_conv5, outputs_pool5, outputs_conv6 = model(features_conv5, features_pool5, features_conv6)
            loss_conv5 = criterion(outputs_conv5, labels)
            loss_pool5 = criterion(outputs_pool5, labels)
            loss_conv6 = criterion(outputs_conv6, labels)
            total_loss = (loss_pool5 + loss_conv6 + loss_conv5) / 3.0
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * features_pool5.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        correct = 0  # Correctly predicted samples
        total = 0    # Total number of samples

        with torch.no_grad():
            for features_conv5, features_pool5, features_conv6, labels in val_loader:
                features_conv5, features_pool5, features_conv6, labels = features_conv5.to(device), features_pool5.to(device), features_conv6.to(device), labels.to(device)
                outputs_conv5, outputs_pool5, outputs_conv6 = model(features_conv5, features_pool5, features_conv6)
                loss_conv5 = criterion(outputs_conv5, labels)
                loss_pool5 = criterion(outputs_pool5, labels)
                loss_conv6 = criterion(outputs_conv6, labels)
                total_val_loss = (loss_conv5 + loss_pool5 + loss_conv6) / 3.0
                val_loss += total_val_loss.item() * features_pool5.size(0)

                # Accuracy computation
                _, predicted_conv5 = torch.max(outputs_conv5.data, 1)
                _, predicted_pool5 = torch.max(outputs_pool5.data, 1)
                _, predicted_conv6 = torch.max(outputs_conv6.data, 1)
                total_predictions = (predicted_pool5 + predicted_conv6 + predicted_conv5) / 3.0
                total_predictions = torch.round(total_predictions)
                total += labels.size(0)
                correct += (total_predictions == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total  # Calculating accuracy percentage
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), args.output_file)

    print('MLP model saved successfully')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir")
    parser.add_argument("feat_dim", type=int)
    parser.add_argument("list_videos")
    parser.add_argument("output_file")
    parser.add_argument("--feat_appendix", default=".csv")

    args = parser.parse_args()
    main(args)