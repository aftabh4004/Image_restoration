import torch
from tqdm import tqdm



def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    model.train()
    
    progress_bar = tqdm(total=len(data_loader), desc='Training')

    train_loss = 0
    count = 0
    for i, (masked_img, unmasked_img) in enumerate(data_loader):
        masked_img = masked_img.to(device)
        unmasked_img = unmasked_img.to(device)

        optimizer.zero_grad()
        reconstructed, mu, log_var = model(masked_img)

        kl = -0.5 * torch.sum(1 + log_var 
                                      - mu**2 
                                      - torch.exp(log_var), 
                                      axis=1) # sum over latent dimension

        reconstruction_loss_factor = 1000
        reconstruction_loss = torch.mean((unmasked_img - reconstructed) ** 2, dim=[1, 2, 3])
        reconstruction_loss = reconstruction_loss_factor * reconstruction_loss
        

        # for mnist dataset
        # reconstruction_loss_factor = 1000
        # reconstruction_loss = torch.mean((masked_img - reconstructed) ** 2, dim=[1, 2, 3])
        # reconstruction_loss = reconstruction_loss_factor * reconstruction_loss
        

        kl = kl.mean()
        reconstruction_loss =  reconstruction_loss.mean()
        
        loss = reconstruction_loss + kl
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        count += 1

        progress_bar.set_postfix_str(f'LR: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {(train_loss/count):.4f}', refresh=False)
        progress_bar.update()

        
    train_loss /= count
    # print(f"Epoch[{epoch}] lr: {optimizer.param_groups[0]['lr']} train loss {train_loss}")


def evaluate(model, criterion, data_loader, device):
    model.eval()
    progress_bar = tqdm(total=len(data_loader), desc='Validation')
    val_loss = 0
    count = 0
    with torch.no_grad():
        for i, (masked_img, unmasked_img) in enumerate(data_loader):
            masked_img = masked_img.to(device)
            unmasked_img = unmasked_img.to(device)

            
            reconstructed, mu, log_var = model(masked_img)

            kl = -0.5 * torch.sum(1 + log_var 
                                        - mu**2 
                                        - torch.exp(log_var), 
                                        axis=1) # sum over latent dimension

            reconstruction_loss_factor = 1000
            reconstruction_loss = torch.mean((unmasked_img - reconstructed) ** 2, dim=[1, 2, 3])
            reconstruction_loss = reconstruction_loss_factor * reconstruction_loss
            

            # for mnist dataset
            # reconstruction_loss_factor = 1000
            # reconstruction_loss = torch.mean((masked_img - reconstructed) ** 2, dim=[1, 2, 3])
            # reconstruction_loss = reconstruction_loss_factor * reconstruction_loss
            

            kl = kl.mean()
            reconstruction_loss =  reconstruction_loss.mean()
            
            loss = reconstruction_loss + kl
            
            
            # print(count, log_var, mu, kl)
            
           
            
            val_loss += loss.item()
            count += 1

            progress_bar.set_postfix_str(f'Loss: {(val_loss/count):.4f}', refresh=False)
            progress_bar.update()


    val_loss /= count
    return val_loss