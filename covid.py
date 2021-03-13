import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.metrics import Accuracy
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

# NOTE: code below written for use in Python / Cluster
# We install:
#   * Pytorch Lightning Version 1.2.1 (https://github.com/PyTorchLightning/pytorch-lightning)
#   * torch Version 1.7.1 (https://github.com/pytorch/pytorch)
#   * torchvision Version 0.8.2 (https://github.com/pytorch/vision)
#   * pathlib Version 1.0.1 (https://pathlib.readthedocs.io/en/pep428/)
#   * pillow Version 7.0.0 (https://github.com/python-pillow/Pillow/blob/88bd672dafad68b419ea29bef941dfa17f941038/docs/installation.rst)

HOME = os.environ.get("HOME")
BASE = Path(str(HOME) + "/projects/def-jlevman/x2019/covid")
ROOT = BASE / "images"

COVID_TEXT_LABELS = Path(str(BASE) + "/covid_text_labels")
COVID_TRAIN = COVID_TEXT_LABELS / "covid_train_labels.txt"
COVID_VAL = COVID_TEXT_LABELS / "covid_val_labels.txt"
COVID_TEST = COVID_TEXT_LABELS / "covid_test_labels.txt"

NONCOVID_TEXT_LABELS = Path(str(BASE) + "/noncovid_text_labels")
NONCOVID_TRAIN = NONCOVID_TEXT_LABELS / "noncovid_train_labels.txt"
NONCOVID_VAL = NONCOVID_TEXT_LABELS / "noncovid_val_labels.txt"
NONCOVID_TEST = NONCOVID_TEXT_LABELS / "noncovid_test_labels.txt"

BATCH_SIZE = 8
EPOCHS = 500


INPUT_SIZE = (224, 224)


def read_text_labels(text_path: str) -> List[str]:
    """read file in `txt_path`, stripping away leading and trailing whitespace"""
    with open(text_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


class CovidDataset(Dataset):
    def __init__(self, root_dir, text_COVID, text_NonCOVID, transform):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.text_path = [text_COVID, text_NonCOVID]
        self.classes = ["CT_COVID", "CT_NonCOVID"]
        self.covid_images = read_text_labels(text_COVID)
        self.noncovid_images = read_text_labels(text_NonCOVID)
        self.covid_path = Path(root_dir).resolve() / "CT_COVID"
        self.noncovid_path = Path(root_dir).resolve() / "CT_NonCOVID"
        self.covid_images = [self.covid_path /
                             img for img in self.covid_images]
        self.noncovid_images = [self.noncovid_path /
                                img for img in self.noncovid_images]
        self.images = self.covid_images + self.noncovid_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        img_path = self.images[index]
        parent = img_path.parent.name
        label = 1 if parent == "CT_COVID" else 0

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def covid_dataset_helper():
    def get_transforms(input_size: int = INPUT_SIZE):

        # The mean and std values is suggested by torchvision.model documentation
        # link: https://pytorch.org/vision/0.8/models.html
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose(
            [

                # transforms.Resize(256),
                # transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # transforms.ToTensor(),
                # normalize,
                transforms.Resize(256),
                transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

        val_transform = transforms.Compose(
            [

                # transforms.Resize((224, 224)),
                # transforms.ToTensor(),
                # normalize,
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return train_transform, val_transform

    train_transform, val_transform = get_transforms()
    trainset = CovidDataset(
        root_dir=str(ROOT),
        text_COVID=str(COVID_TRAIN),
        text_NonCOVID=str(NONCOVID_TRAIN),
        transform=train_transform,
    )
    valset = CovidDataset(
        root_dir=str(ROOT),
        text_COVID=str(COVID_VAL),
        text_NonCOVID=str(NONCOVID_VAL),
        transform=val_transform,
    )
    testset = CovidDataset(
        root_dir=str(ROOT),
        text_COVID=str(COVID_TEST),
        text_NonCOVID=str(NONCOVID_TEST),
        transform=val_transform,
    )

    return trainset, valset, testset


class ResnetTransferLearning(LightningModule):
    """
    Lightning Module is adapted from https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction_guide.html
    """
    def __init__(self, input_shape, learning_rate=1e-5):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.accuracy = pl.metrics.Accuracy()
        # Initializing a pretrained resnet18 model.
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.eval()
        """ we are using the convolutional output of each model, 1000 is the convolutional output of resnet18 """
        self.classfier = nn.Linear(1000, 1)

    def forward(self, x):
        self.feature_extractor.eval()
        x = self.feature_extractor(x)
        x = x.unsqueeze(1)
        x = self.classfier(x)
        x = x.squeeze()
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        output = self(x)
        criterion = BCEWithLogitsLoss()
        try:
            loss = criterion(output.unsqueeze(1), y.unsqueeze(1).float())
        except BaseException as e:

            raise e

        logits = torch.sigmoid(output)
        pred = torch.round(logits)
        acc = torch.mean((pred == y).float())
        self.log('train_acc_step', acc, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        criterion = BCEWithLogitsLoss()
        logits = torch.sigmoid(output)
        pred = torch.round(logits)
        acc = torch.mean((pred == y).float())
        self.log('val_acc', acc, prog_bar=True)
        loss = criterion(output.unsqueeze(1), y.unsqueeze(1).float())
        self.log('val_loss', loss)
        # self.valid_acc(logits, y)
        # self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        criterion = BCEWithLogitsLoss()
        logits = torch.sigmoid(output)
        pred = torch.round(logits)
        self.test_pred.append(pred)
        acc = torch.mean((pred == y).float())
        self.log('test_acc', acc, prog_bar=True)
        loss = criterion(output.unsqueeze(1), y.unsqueeze(1).float())
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def bagging_loader(dataset: CovidDataset, percent: float = 0.6, shuffle: bool = False) -> DataLoader:
    size = int(round(percent * len(dataset)))

    # idx = torch.randperm(size)
    idx = random.sample(range(0, len(dataset)), size)

    return DataLoader(dataset, sampler=SubsetRandomSampler(idx), batch_size=BATCH_SIZE, drop_last=True, shuffle=shuffle)


def generatebaggingModels(model_num):
    model_list = []

    model = ResnetTransferLearning((3, 224, 224), 2)
    CHECKPOINT_ROOT = Path(
        str(HOME) + "/projects/def-jlevman/x2019/covid/checkpoints")

    for n in range(model_num):
        bag_train_loader = bagging_loader(trainset)
        bag_val_loader = bagging_loader(valset)
        CHECKPOINT_PATH = CHECKPOINT_ROOT / str(n)
        checkpoint_callback = ModelCheckpoint(CHECKPOINT_PATH)
        trainer = pl.Trainer(gpus=2,num_nodes=2,accelerator='ddp', callbacks=checkpoint_callback,
                             max_epochs=EPOCHS, progress_bar_refresh_rate=5)
        trainer.fit(model, bag_train_loader, bag_val_loader)
        best_model = ResnetTransferLearning.load_from_checkpoint(
            checkpoint_callback.best_model_path)
        model_list.append(best_model)
    return model_list


def bagging(model_num):
    model_list = []
    model_list = generatebaggingModels(model_num)
    pred_dic = {}

    for model in model_list:
        model.eval()
        model.freeze()
        pred_tensor = torch.tensor([])
        labels_tensor = torch.tensor([])

        for (input, labels) in test_loader:
            output = model(input)
            logits = torch.sigmoid(output)
            pred = torch.round(logits)
            pred_tensor = torch.cat((pred_tensor, pred))
            labels_tensor = torch.cat((labels_tensor, labels))
            pred_dic[model_list.index(model)] = pred_tensor

    value = pred_dic.values()
    total = sum(value)
    pred_list = torch.round((total/model_num))
    acc = torch.mean((pred_list == labels_tensor).float())
    return acc


if __name__ == "__main__":
    trainset, valset, testset = covid_dataset_helper()
    train_loader = DataLoader(
        trainset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE,
                            drop_last=True, shuffle=False)
    test_loader = DataLoader(
        testset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
    bag_train_loader = bagging_loader(trainset)
    bag_val_loader = bagging_loader(valset)

    acc = bagging(11)
    print("acc: ",acc)
