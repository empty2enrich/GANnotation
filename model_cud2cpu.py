
import torch
from model import Generator

def convert_cpu():
    path = ''
    gen = Generator()
    net_weight = torch.load(path, map_location='cpu')
    net_dict = {k.replace('module.',''): v for k,v in net_weight['state_dict'].items()}
    gen.load_state_dict(net_dict)
    torch.save(gen.state_dict(), 'model.pkl')

def load_cpu():
    gen = Generator()
    gen.load_state_dict(torch.load('model.pkl'))
    print("load finish!")

if __name__ == "__main__":
    convert_cpu()

