import torch
import torch.nn as nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self, z_dim, h_dim=200, channel=3, img_size=128):
        super(VariationalAutoEncoder, self).__init__()
        self.flat_dim = 256 * 16 * 16
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 32, 3, stride=1, padding=1),
            nn.MaxPool2d((2,2), stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # nn.Dropout2d(0.25),
            
             
            nn.Conv2d(32, 64, 3,stride=1, padding=1),
            nn.MaxPool2d((2,2), stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # nn.Dropout2d(0.25),
            
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.MaxPool2d((2,2), stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # nn.Dropout2d(0.25),
            
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.MaxPool2d((2,2), stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # nn.Dropout2d(0.25),
            
            
            
            nn.Flatten(),
            nn.Linear(self.flat_dim, h_dim)
        )
        
        self.hid_to_mu = nn.Sequential(
            nn.Linear(h_dim, z_dim),
        )
        
        self.hid_to_sigma = nn.Sequential(
            nn.Linear(h_dim, z_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, self.flat_dim),
            nn.Unflatten(1, (256, 16, 16)),
            
            
            
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Upsample((32, 32)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Upsample((64, 64)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Upsample((128, 128)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Upsample((256, 256)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(3),
            
            # nn.Tanh()
        )

    
    def forward(self, x):
        
        #encode
        x_normalized = x
        endcoded = self.encoder(x_normalized)
        
        
        mu = self.hid_to_mu(endcoded)
        log_var = self.hid_to_sigma(endcoded)
        
        #Sampling
        eps = torch.randn(mu.size(0), mu.size(1)).to(mu.get_device())
        z = mu + eps * torch.exp(log_var/2.)
        
        #decode
        
        reconstructed = self.decoder(z)
       
        
        return reconstructed, mu, log_var