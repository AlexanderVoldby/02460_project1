import torch
from torchvision import datasets, transforms

class MnistDataLoader:
    def __init__(self, args):
        self.args = args

        data = datasets.MNIST('data/', train =True, download =True,
                transform = transforms.Compose ([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
                transforms.Lambda(lambda x: x.flatten())
                ]))
        self.train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)

    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
        