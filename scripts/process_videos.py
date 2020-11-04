import os
import pickle
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import cv2 as cv
import numpy as np
from tqdm.std import tqdm

from get_descriptors_for_images import descriptors_to_test, test_image

descriptors_to_test.update({"SVC": None})

size = 2048
train_ppe = f'../data/ppe/resized/photos_{size}/item_pics/20200918_161314.jpg'
train_zvv = f'../data/zvv/resized/item_pics/photos_{size}/P00930-164836_1.jpg'
train_rvv = f'../data/rvv/resized/item_pics/photos_{size}/IMG_20201003_153648.jpg'

classes = ["girl", "tt_racket", "cooler", "unknown"]

models_path = "../models/{}/model_{}"
scalers_path = "../scalers/{}.pkl"

best_models = {
    'ORB(nfeatures=500, scoreType=ORB_HARRIS_SCORE)': '1604414323.641065',
    'ORB(nfeatures=500, scoreType=ORB_FAST_SCORE)': '1604414483.1804457',
    'ORB(nfeatures=250, scoreType=ORB_HARRIS_SCORE)': '1604414917.1122782',
    'BRISK(thresh=10)': '1604415232.651425',
    'BRISK(thresh=20)': '1604415724.5486772',
    'BRISK(thresh=25)': '1604416024.630207',
    'BRISK(thresh=30)': '1604416181.9278598',
    'BRISK(thresh=40)': '1604416651.7820601',
    'BRIEF()': '1604416811.7798843'
}


class Model(nn.Module):

    def __init__(self, hidden_size, input_size, layers=1, mode="hidden"):
        super(Model, self).__init__()
        self.mode = mode
        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=layers, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(hidden_size * 2 * layers, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc_out = nn.Linear(hidden_size, 4)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        _, hidden_rnn = self.rnn(x)
        if self.mode == "hidden":
            out = hidden_rnn[0].transpose(0, 1)
            out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
        elif self.mode == "cells":
            out = hidden_rnn[1].transpose(0, 1)
            out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
        else:
            raise ValueError("Unknown `mode` parameter.")
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.bn1(out)
        return self.fc_out(out)

    def predict(self, x):
        out = self(x)
        return self.softmax(out)


def preprocess_image(input_image):
    input_image = Image.fromarray(input_image)
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0)


def get_trained_descriptor(descriptor, train_img_path):
    train_img = cv.imread(train_img_path, cv.IMREAD_GRAYSCALE)

    if not isinstance(descriptor, cv.xfeatures2d_BriefDescriptorExtractor):
        train_kp, train_descriptors = descriptor.detectAndCompute(
            train_img, None)
    else:
        # needed for BRIEF descriptor
        star = cv.xfeatures2d.StarDetector_create()
        star_kp = star.detect(train_img, None)
        train_kp, train_descriptors = descriptor.compute(
            train_img, star_kp)
    return train_descriptors


def load_lstm(descriptor_name):
    params = torch.load(models_path.format(descriptor_name, best_models[descriptor_name]))
    model = Model(**params["params"])
    model.load_state_dict(params["state_dict"])

    model = model.eval().cuda()

    with open(scalers_path.format(descriptor_name), "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


def load_svc(path):
    model = nn.Sequential(*list(
        torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl').children()
    )[:-1]).to('cuda')

    with open(path + "transformer.pkl", "rb") as file:
        transformer = pickle.load(file)

    with open(path + "svc.pkl", "rb") as file:
        svc = pickle.load(file)

    return model, transformer, svc


for descriptor_name, descriptor in tqdm(descriptors_to_test.items()):
    os.makedirs(f"../results/videos/{descriptor_name}", exist_ok=True)
    if descriptor is None:
        model, transformer, svc = load_svc("../")
    else:
        model, scaler = load_lstm(descriptor_name)
    for video_name, ideal in [('ppe', train_ppe), ('zvv', train_zvv), ('rvv', train_rvv)]:
        cap = cv.VideoCapture(f'../data/videos/{video_name}.mp4')
        fourcc = cv.VideoWriter_fourcc(*'MP4V')
        writer = cv.VideoWriter(
            f'../results/videos/{descriptor_name}/{video_name}.mp4', fourcc, 25.0, (1920, 1080))

        if descriptor is not None:
            train_descriptor = get_trained_descriptor(descriptor, ideal)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            if descriptor is None:
                img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                out = preprocess_image(img).cuda()
                out = model(out).detach().cpu().squeeze().numpy()
                out = transformer.transform(out.reshape(1, -1))
                out = svc.predict(out).ravel()
            else:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                res = test_image(train_descriptor,  descriptor, gray, from_frame=True)
                out = [3]
                if res.shape[0] > 0:
                    out = nn.utils.rnn.pack_sequence([
                        torch.tensor(scaler.transform(res).astype(np.float32))
                    ], enforce_sorted=False).cuda()
                    out = model.predict(out).detach().cpu().numpy().argmax(1).ravel()

            cv.putText(frame, f'class: {classes[out[0]]}', (10, 200), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

            writer.write(frame)
        writer.release()
        cap.release()
