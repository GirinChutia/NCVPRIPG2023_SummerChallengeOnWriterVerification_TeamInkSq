from modelling.model import SiameseNetwork_SV
import torch, glob
from torchvision import transforms
from PIL import Image
import pandas as pd
import os, gc
from tqdm import tqdm
import argparse


def perform_evaluation(csv_path, img_dir, model_path, set_name='val'):

    feed_shape = [3, 200, 2600]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(feed_shape[1:], antialias=True),
        ]
    )

    backbone = "resnet50"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork_SV(backbone=backbone)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model = model.eval()

    val_df = pd.read_csv(csv_path)

    IMG1_NAME, IMG2_NAME, LABELS, _preds = [], [], [], []
    IMG1_IMG2_NAME = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        
        img1_path = os.path.join(img_dir, row["img1_name"])
        img2_path = os.path.join(img_dir, row["img2_name"])
        
        if set_name == 'val':
            IMG1_NAME.append(row["img1_name"])
            IMG2_NAME.append(row["img2_name"])
            LABELS.append(row["label"])
            
        if set_name == 'test':
            IMG1_IMG2_NAME.append(os.path.basename(row["img1_name"]) + '_' + os.path.basename(row["img2_name"]))
            
        p = infer_images(
            model, img1_path, img2_path, transform=transform, device=device
        )

        p = p.detach().cpu().numpy()[0][0]
        p = round(p, 5)
        _preds.append(p)

    if set_name == 'val':
        pred_df = pd.DataFrame.from_dict(
            {
                "img1_name": IMG1_NAME,
                "img2_name": IMG2_NAME,
                "label": LABELS,
                "pred": _preds,
            }
        )
        
        _preds_thresh = []
        for i in _preds:
            if i > 0.5:
                _preds_thresh.append(1)
            else:
                _preds_thresh.append(0)

        submission_df = pd.DataFrame.from_dict(
            {
                "img1_name": IMG1_NAME,
                "img2_name": IMG2_NAME,
                "label": _preds_thresh,
                "proba": _preds,
            }
        )
    
    if set_name == 'test':
        pred_df = pd.DataFrame.from_dict(
            {
                "id": IMG1_IMG2_NAME,
                "proba": _preds,
            }
        )
        submission_df = pred_df
        
    return pred_df, submission_df


def infer_images(model, image_path1, image_path2, transform, device):
    image1, image2 = (
        Image.open(image_path1).convert("RGB"),
        Image.open(image_path2).convert("RGB"),
    )
    image1, image2 = transform(image1).float(), transform(image2).float()
    image1, image2 = image1.unsqueeze(0), image2.unsqueeze(0)
    image1, image2 = map(lambda x: x.to(device), [image1, image2])
    prob = model(image1, image2)
    del image1, image2
    torch.cuda.empty_cache()
    gc.collect()
    return prob


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, required=True, help="path to the test csv file"
    )
    parser.add_argument(
        "--img_dir", type=str, required=True, help="path to the test image folder"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="path to the saved model (.pth)"
    )
    parser.add_argument(
        "--submission_csv",
        type=str,
        required=True,
        help="path of the submission csv where the output submission will be saved",
    )
    parser.add_argument(
        "--set_name",
        type=str,
        required=True,
        help="Set for evaluation : val or test ",
    )
    args = parser.parse_args()
    
    pred_df, submission_df = perform_evaluation(csv_path=args.csv_path,
                                                img_dir=args.img_dir,
                                                set_name=args.set_name,
                                                model_path=args.model_path)
    # 'InkSq_02_model2.csv'
    submission_df.to_csv(args.submission_csv, index=False)
    print(f"Submission CSV is saved at : {args.submission_csv}")
