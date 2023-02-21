import torch


class LinexLoss(torch.nn.Module):
    def __init__(self, a):
        super(LinexLoss, self).__init__()
        self.a = a

    def forward(self, output, target):
        self.a = float(self.a)
        loss = (1 + target) * (torch.exp(self.a * target * (1 - output)) - self.a * target * (1 - output) - 1) + (
                    1 - target) * (torch.exp(self.a * target * output) - self.a * target * output - 1)
        return torch.mean(loss)
