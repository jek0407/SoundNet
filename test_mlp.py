import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from train_mlp import MultiHeadClassifier, AudioDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file")
    parser.add_argument("feat_dir")
    parser.add_argument("feat_dim", type=int)
    parser.add_argument("list_videos")
    parser.add_argument("output_file")
    parser.add_argument("--file_ext", default=".csv")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    feat_list_conv5, feat_list_pool5, feat_list_conv6 = [], [], []
    video_ids = []

    for line in open(args.list_videos).readlines()[1:]:
        video_id = line.strip().split(",")[0]
        video_ids.append(video_id)

        feat_filepath_conv5 = os.path.join(args.feat_dir, "conv5", video_id + args.file_ext)
        feat_filepath_pool5 = os.path.join(args.feat_dir, "pool5", video_id + args.file_ext)
        feat_filepath_conv6 = os.path.join(args.feat_dir, "conv6", video_id + args.file_ext)

        # If features don't exist, append zeros
        feat_list_conv5.append(np.genfromtxt(feat_filepath_conv5, delimiter=";", dtype="float") if os.path.exists(feat_filepath_conv5) else np.zeros(args.feat_dim))
        feat_list_pool5.append(np.genfromtxt(feat_filepath_pool5, delimiter=";", dtype="float") if os.path.exists(feat_filepath_pool5) else np.zeros(args.feat_dim))
        feat_list_conv6.append(np.genfromtxt(feat_filepath_conv6, delimiter=";", dtype="float") if os.path.exists(feat_filepath_conv6) else np.zeros(args.feat_dim))

    conv5_dim = len(feat_list_conv5[0])
    pool5_dim = len(feat_list_pool5[0])
    conv6_dim = len(feat_list_conv6[0])
    num_classes = 15  # You can adjust this accordingly if needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadClassifier(conv5_dim, pool5_dim, conv6_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(args.model_file))
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for conv5_feat, pool5_feat, conv6_feat in zip(feat_list_conv5, feat_list_pool5, feat_list_conv6):
            conv5_feat = torch.tensor([conv5_feat], dtype=torch.float).to(device)
            pool5_feat = torch.tensor([pool5_feat], dtype=torch.float).to(device)
            conv6_feat = torch.tensor([conv6_feat], dtype=torch.float).to(device)
            
            outputs_conv5, outputs_pool5, outputs_conv6 = model(conv5_feat, pool5_feat, conv6_feat)
            
            # Averaging the outputs
            total_predictions = (outputs_pool5 + outputs_conv6 + outputs_conv5) / 3.0
            _, predicted = torch.max(total_predictions.data, 1)
            predictions.append(predicted.item())

    with open(args.output_file, "w") as f:
        f.writelines("Id,Category\n")
        for i, pred_class in enumerate(predictions):
            f.writelines("%s,%d\n" % (video_ids[i], pred_class))

    print("Predictions saved to", args.output_file)