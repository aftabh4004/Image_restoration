#Imports

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import  transforms, datasets

from dataset import AnimalDataset
from model import VariationalAutoEncoder
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--z_dim', type=int, default=128, help='Z dimension')
    parser.add_argument('--pre-epoch', type=int, default=20, help='Pretraning epoch')
    parser.add_argument('--epoch', type=int, default=40, help='traning epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--sched', type=str, default='cosine', help='lr sechdular')
    parser.add_argument('--dataset_root', type=str, default='/home/mt0/22CS60R54/image_regen/Dataset', help='Dataset root directory')
    parser.add_argument('--list_root', type=str, default='/home/mt0/22CS60R54/image_regen/dataset_list', help='Dataset lists root directory')

    parser.add_argument('--device', type=str, default='cpu', help='cpu/cuda')
    parser.add_argument('--img_size', type=int, default=128, help='imput image size')

    return parser




def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(args.device)
    transform = transforms.Compose([
        transforms.ToTensor(),         # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
    ])

    # train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    ## Dataloaders
    train_list = os.path.join(args.list_root, "train.txt")
    train_dataset = AnimalDataset(data_list=train_list, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    
    val_list = os.path.join(args.list_root, "test.txt")
    val_dataset = AnimalDataset(data_list=val_list, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    test_list = os.path.join(args.list_root, "val.txt")
    test_dataset = AnimalDataset(data_list=test_list, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    

    model = VariationalAutoEncoder(z_dim=args.z_dim, h_dim=args.hidden_dim, channel=3, img_size=args.img_size)
    # print(model)
    # return
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    print("Start training...")
    
    for epoch in range(args.epoch):
        train_one_epoch(model, criterion, train_loader, optimizer, args.device,  epoch)
        print()
        val_loss = evaluate(model, criterion, val_loader, args.device)
        print()
        scheduler.step(val_loss)
    
    torch.save(model, 'modelcolor.h5')



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)

