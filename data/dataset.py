
from torchvision import transforms as T
import torchvision.datasets as datasets

class MNIST():
    
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        
        self.Dtrain = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        self.Dtest = datasets.MNIST(root='./data', train=False, transform=self.transform, download=True)
        
    def get(self, train=True):
        return self.Dtrain if train else self.Dtest