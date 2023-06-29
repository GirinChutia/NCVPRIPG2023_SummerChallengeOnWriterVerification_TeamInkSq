import gc
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelling.dataset import Dataset
from modelling.model import SiameseNetwork_SV


def train(
    epochs,
    weight_path=None,
    experiment_folder_path="exp",
    learning_rate=0.002,
    save_after=25,
    backbone="resnet50",
    batch_size=64,
    train_dataset_folder="",
    val_dataset_folder="",
    seed=21,
):

    torch.random.manual_seed(seed)
    os.makedirs(experiment_folder_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Dataset(train_dataset_folder, shuffle_pairs=True, augment=True)
    val_dataset = Dataset(val_dataset_folder, shuffle_pairs=False, augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = SiameseNetwork_SV(backbone=backbone)

    if weight_path:
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    writer = SummaryWriter(os.path.join(experiment_folder_path, "summary"))

    best_val = 1000000000000

    for epoch in range(epochs):
        print("[{} / {}]".format(epoch, epochs))
        model.train()
        losses = []
        correct = 0
        total = 0

        for (img1, img2), y, (class1, class2), (imp1, imp2) in tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Train Epoch {epoch}/{epochs}",
        ):

            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

            del img1, img2, prob, loss

        gc.collect()
        torch.cuda.empty_cache()

        writer.add_scalar("train_loss", sum(losses) / len(losses), epoch)
        writer.add_scalar("train_acc", correct / total, epoch)

        print(
            "\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(
                sum(losses) / len(losses), correct / total
            )
        )

        model.eval()

        losses = []
        correct = 0
        total = 0

        for (img1, img2), y, (class1, class2), (imp1, imp2) in tqdm(
            val_dataloader,
            total=len(val_dataloader),
            desc=f"Epoch {epoch}, Validating ..",
        ):
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

            del img1, img2, prob, loss

        gc.collect()
        torch.cuda.empty_cache()

        val_loss = sum(losses) / max(1, len(losses))
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_acc", correct / total, epoch)

        print(
            "\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(
                val_loss, correct / total
            )
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": backbone,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(experiment_folder_path, "best.pth"),
            )

        if (epoch + 1) % save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": backbone,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(experiment_folder_path, "epoch_{}.pth".format(epoch + 1)),
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, required=True, help="Total epochs for training"
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        required=False,
        help="Pretrained weights path (Optional)",
    )
    parser.add_argument(
        "--experiment_folder_path",
        type=str,
        required=True,
        default="exp0",
        help="Experiment folder path (for saving logs and weights)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.002,
        required=True,
        help="learning_rate",
    )
    parser.add_argument(
        "--save_after",
        type=int,
        required=False,
        default=10,
        help="Weight save frequency",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        required=False,
        help="Model backbone",
    )
    parser.add_argument(
        "--batch_size", type=str, default=64, required=True, help="batch_size",
    )
    parser.add_argument(
        "--train_dataset_folder",
        type=str,
        required=True,
        help="training dataset folder",
    )
    parser.add_argument(
        "--val_dataset_folder",
        type=str,
        required=True,
        help="validation dataset folder",
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=21, help="seed",
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        weight_path=args.weight_path,
        experiment_folder_path=args.experiment_folder_path,
        learning_rate=args.learning_rate,
        save_after=args.save_after,
        backbone=args.backbone,
        batch_size=args.batch_size,
        train_dataset_folder=args.train_dataset_folder,
        val_dataset_folder=args.val_dataset_folder,
        seed=args.seed,
    )
