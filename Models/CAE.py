import torch 
import torch.nn as nn
from torch.optim import Adam

class ConvAutoencoder(nn.Module):
    """
    This class implements the ConvAutoencoder for the feature extraction Purpose.
    First we Train the CAE(Convolutional Autoencoder) for reconstruction of input data,
    based upon the MAE loss and Adam Optimizer. 
    The we seperate the Encoder from the Decoder and use the encoder for 
    feature extraction purpose in Vision Transformers.
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def extract_features(self, x):
        """Extract features using the encoder part only."""
        with torch.no_grad():
            return self.encoder(x)


# ------------------> Example usage <------------------

def train_autoencoder(model, data_loader, epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for data in data_loader:
            inputs = data[0]  # Assuming data is a tuple (image, label)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Feature extraction
def extract_features(model, data):
    model.eval()
    return model.extract_features(data)
