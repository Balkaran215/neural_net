import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline  

# Create a Model Class that inherits nn.module 
class Model(nn.Module):
    # input layer (4 layers of the flower) -> (shoot to hidden layer 1) -> (Hidden layer 2) -> (output)
    def __innit__(self, in_features = 4, h1 = 8, h2 = 9, h3 = 8, output_feature = 3):
        super().__innit__() # instantiate our nn.Module
        self.fc1 = nn.linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.output = nn.linear(h3, output_feature)
    
    def move(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)

        return x

torch.manual_seed(41)

model = Model()

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)
print(my_df)