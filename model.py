#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Insup Lee <islee94@korea.ac.kr>
# July 2020

"""
# https://nyan101.github.io/blog/notes-on-pytorch-01
# https://tutorials.pytorch.kr/
# GAN 모델링 하기
- 네트워크 구조 만들기 
- loss 함수 및 optimizer 정의하기 
- Discriminator 학습
	+ loss 값 계산
	+ 미분 값 초기화 및 새로운 미분 값 계산
	+ 최적화 기법을 통해 매개변수 업데이트
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import logging_time


def make_dataloader():
    # 데이터 전처리 방식 지정
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert data to Tensor of pytorch
        transforms.Normalize(mean=(0.5,), std=(0.5,))  # 픽셀 값 0~1 -> -1~1
    ])

    # MNIST 데이터셋을 불러온다.
    mnist = datasets.MNIST(root='data', download=True, transform=transform)

    # 데이터를 한번에 batch_size만큼 가져오는 dataloader 생성
    dataloader = DataLoader(mnist, batch_size=60, shuffle=True)

    return dataloader


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self._inplace = False
        self.main = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            nn.LeakyReLU(0.2, inplace=self._inplace),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2, inplace=self._inplace),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2, inplace=self._inplace),
            nn.Linear(in_features=1024, out_features=28 * 28),
            nn.Tanh()  # TODO 이건 뭐지?
        )

    # (batch_size x 100) 크기의 랜덤 벡터를 받아 이미지를 (batch_size x 1 x 28 x 28) 크기로 출력
    def forward(self, input):
        img = self.main(input).view(-1, 1, 28, 28)  # tensor의 모양을 변경 (resize하고 플 대 torch.view 사용)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()  # nn.module인가
        self._inplace = False
        self.main = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=1024),
            nn.LeakyReLU(0.2, inplace=self._inplace),  # 앞은 negative slope, inplace는?
            nn.Dropout(inplace=self._inplace),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2, inplace=self._inplace),
            nn.Dropout(inplace=self._inplace),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2, inplace=self._inplace),
            nn.Dropout(inplace=self._inplace),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    # (batch_size x 1 x 28 x 28) 크기의 이미지를 받아, 이미지가 진짜일 확률 0~1사이 출력
    def forward(self, input):
        validity = self.main(input).view(-1, 28 * 28)
        return validity


@logging_time
def training(epochs=3):
    G = Generator()
    D = Discriminator()

    dataloader = make_dataloader()
    criterion = nn.BCELoss()  # Binary Cross Entropy loss

    G_opt = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))  # Generator의 params를 최적화하는 optimizer
    D_opt = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        print(".. epoch {}".format(epoch+1))
        # 한번에 batch_size 만큼 데이터를 가져옴
        for real_data, _ in dataloader:
            batch_size = real_data.size(0)
            real_data = Variable(real_data)  # 데이터를 pytorch의 변수로 변환

            # Discriminator training
            target_real = Variable(torch.ones(batch_size, 1))  # 정답지에 해당되는 변수 만듬
            target_fake = Variable(torch.zeros(batch_size, 1))

            D_result_from_real = D(real_data)  # 진짜 이미지를 D에 넣음
            D_loss_real = criterion(D_result_from_real, target_real)

            z = Variable(torch.randn((batch_size, 100)))  # 생성자 입력 랜덤 벡터 Z 생성
            fake_data = G(z)  # 생성자로 가짜 이미지 생성

            D_result_from_fake = D(fake_data)
            D_loss_fake = criterion(D_result_from_fake, target_fake)

            D_loss = D_loss_real + D_loss_fake
            D.zero_grad()  # D의 매개변수의 미분값을 0으로 초기화
            D_loss.backward()  # 역전파를 통해 매개변수의 loss에 대한 미분값 계산
            D_opt.step()  # 최적화 기법 이용, D의 매배변수 업데이트

            # Generator traininng
            z = Variable(torch.randn((batch_size, 100)))
            # z = z.cuda()  # GPU 있을때 할 수 있음


            fake_data = G(z)

            D_result_from_fake = D(fake_data)

            G_loss = criterion(D_result_from_fake, target_real)  # G_loss_fake == G_loss
            G.zero_grad()
            G_loss.backward()
            G_opt.step()


if __name__ == '__main__':
    training()
