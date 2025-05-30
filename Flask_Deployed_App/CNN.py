import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            # Conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            # Conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            # Conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)  # Flatten
        out = self.dense_layers(out)
        return out


# Optional: index to class mapping
idx_to_classes = {
    i: name for i, name in enumerate([
        "Aloevera", "Amla", "Amruta_Balli", "Arali", "Ashoka", "Ashwagandha", "Avacado",
        "Bamboo", "Basale", "Betel", "Betel Nut", "Brahmi", "Castor", "Curry_Leaf",
        "Doddapatre", "Ekka", "Ganike", "Gauva", "Geranium", "Henna", "Hibiscus", "Honge",
        "Insulin", "Jasmine", "Lemon", "Lemon_grass", "Mango", "Mint", "Nagadali", "Neem",
        "Nithyapushpa", "Nooni", "Pappaya", "Pepper", "Pomegranate", "Raktachandini",
        "Rose", "Sapota", "Tulasi", "Wood_sorel"
    ])
}
