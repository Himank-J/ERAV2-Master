import torch
from tqdm import tqdm
import torch.nn as nn
import albumentations as A
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets
from torch_lr_finder import LRFinder
from torchvision.utils import make_grid
from albumentations.pytorch import ToTensorV2

class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

class LoadDataset():

    def __init__(self):
        self.__SEED = 1
        self.__cuda = torch.cuda.is_available()
        print("CUDA Available?", self.__cuda)
        self.__mps = torch.backends.mps.is_available()
        print("MPS Available?", self.__mps)

    def getTransforms(self):
        train_transforms = A.Compose(
            [
                A.RandomCrop(32, 32, p=4),
                A.CoarseDropout(max_holes=1,max_height=16,max_width=16,min_holes=1,min_height=16,min_width=16,fill_value=(0.4914, 0.4822, 0.4465),always_apply=False,p=0.5,),
                A.Normalize((0.49139968,0.48215841,0.44653091), (0.24703223,0.24348513,0.26158784)),
                ToTensorV2(),
            ]
        )

        test_transforms = A.Compose(
            [
                A.Normalize((0.49139968,0.48215841,0.44653091), (0.24703223,0.24348513,0.26158784)),
                ToTensorV2(),
            ]
        )
        return train_transforms,test_transforms
  
    def getData(self,batch_size):

        train_transforms,test_transforms = self.getTransforms()
        train = Cifar10SearchDataset('./data', train=True, download=True, transform=train_transforms)
        test = Cifar10SearchDataset('./data', train=False, download=True, transform=test_transforms)
        print('Train and Test data loaded')

        if self.__cuda:
            torch.cuda.manual_seed(self.__SEED)
        elif self.__mps:
            torch.mps.manual_seed(self.__SEED)
        else:
            torch.manual_seed(self.__SEED)

        dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if (self.__cuda or self.__mps) else dict(shuffle=True, batch_size=64)
        train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
        test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
        return train_loader,test_loader
    
class VisualizeData():

    def __init__(self):
        pass

    def visualize(self,train_loader):
        for images, labels in train_loader:
            print('images.shape:', images.shape)
            print('labels.shape:', labels.shape)

            plt.figure(figsize=(15,15))
            plt.axis('off')
            plt.imshow(make_grid(images, nrow=30).permute(1, 2, 0))
            break

class LearningRateFinder():

    def __init__(self):
        self.__cuda = torch.cuda.is_available()
        self.__mps = torch.backends.mps.is_available()
        
    def findLR(self,model,train_loader):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-7, weight_decay=1e-2)
        
        if self.__cuda:
            self.__lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        elif self.__mps:
            self.__lr_finder = LRFinder(model, optimizer, criterion, device="mps")
        else:
            self.__lr_finder = LRFinder(model, optimizer, criterion, device="cpu")

        self.__lr_finder.range_test(train_loader, end_lr=100, num_iter=100, step_mode="exp")
    
    def visualizeLR(self):

        self.__lr_finder.plot()
        self.__lr_finder.reset()

class TrainTest():

    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def GetCorrectPredCount(self, pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

    def train(self, model, device, train_loader, optimizer, criterion):

        model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            pred = model(data)

            loss = criterion(pred, target)
            train_loss+=loss.item()

            loss.backward()
            optimizer.step()

            correct += self.GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        self.train_acc.append(100*correct/processed)
        self.train_losses.append(train_loss/len(train_loader))
        return self.train_acc,self.train_losses
    
    def test(self, model, device, test_loader, criterion):

        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                correct += self.GetCorrectPredCount(output, target)
  
        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        return self.test_acc,self.test_losses
    
    def visualizeLoss(self,train_losses,train_acc,test_losses,test_acc):
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_acc)
        axs[1, 1].set_title("Test Accuracy")
