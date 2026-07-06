"""
This is the CNN model for how the archicture works

We uses PyTorch's library nn to use its CNN classes of conv2d, MaxPool2d, ReLu, AdaptiveAvgPool2d, Linear.
The structure uses 

input (B,1,128,431)
→ conv1 → relu → pool   → (B,32,64,215)
→ conv2 → relu → pool   → (B,64,32,107)
→ gap                   → (B,64,1,1)
→ squeeze(dim=(2,3))    → (B,64)
→ fc                    → (B,20)


"""

from torch import nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(64, 20)

    def forward(self, batch):
        copy = batch
        copy = self.conv1(copy)
        copy = self.relu(copy)
        copy = self.pool(copy)
        copy = self.conv2(copy)
        copy = self.relu(copy)
        copy = self.pool(copy)
        copy = self.gap(copy)
        copy = copy.squeeze(dim=(2, 3))
        copy = self.fc(copy)

        return copy
