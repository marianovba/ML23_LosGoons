import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO: Calcular dimension de salida
        out_dim = self.calc_out_dim(input_dim, kernel_size=3, stride=1, padding=0)

        print("Input dim: ",input_dim)

        # TODO: Define las capas de tu red
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=9)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=8)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64*out_dim*out_dim, 256)
        self.linear2 = nn.Linear(256, n_classes)

        self.to(self.device)
 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        x = x.to(self.device)

        x = self.conv1(x)   
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)

        print(x.shape)

        x = self.flatten(x)

        print(x.shape)

        x = self.linear1(x)
        x = F.relu(x)

        logits = self.linear2(x)
        proba = F.softmax(logits, dim=1)
        return logits, proba
        #return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        '''
            Guarda el modelo en el path especificado
            args:
            - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
            - path (str): path relativo donde se guardará el modelo
        '''
        models_path = file_path / 'models' / model_name
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path / 'modelo1.pth')

    def load_model(self, model_name: str):
        '''
            Carga el modelo en el path especificado
            args:
            - path (str): path relativo donde se guardó el modelo
        '''
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / 'models' / model_name
        self.load_state_dict(torch.load(models_path))