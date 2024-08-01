import numpy as np
import torch
import torch.optim as optim
import src.utils.utils as utils

class Agent():
    # handles the training of the NCA models
    def __init__(self, model,steps=16,channel_n=16,batch_size=16):
        self.model = model
        self.steps = steps
        self.channel_n = channel_n
        self.batch_size=batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_channels = 3
        self.output_channels = 13
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0004, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9999)

    def make_seed(self, img):
        # creates the seed, i.e. padding the image with zeros to the desired number of channels
        seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], self.channel_n), dtype=torch.float32).to(self.device)#torch.from_numpy(np.zeros([img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')], np.float32)).to(self.device)
        seed[..., 0:img.shape[-1]] = img
        return seed

    def prepare_data(self, data, eval=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, targets = data
        inputs = self.make_seed(inputs)
        return inputs,targets.to(device)

    def get_outputs(self, data, **kwargs):
        #compute the output of the model
        inputs, targets = data
        output,feature_map = self.model(inputs, steps=self.steps, fire_rate=0.5)
        return output, targets, feature_map

    def batch_step(self, data, loss_f, train=True):
        #Execute a single batch training step
        data = self.prepare_data(data)
        outputs, targets, _ = self.get_outputs(data)

        #for unet replace the two lines above with:
        #inputs, targets = data
        #inputs=inputs.permute(0,3,1,2).to(self.device)
        #targets.to(self.device)
        #outputs = self.model(inputs.float()).to(self.device)

        self.optimizer.zero_grad()
        loss = 0
        loss_f=loss_f()
        loss = loss_f(outputs.to(self.device),targets.to(self.device))
        
        if train == False:
            return loss.detach()

        if loss != 0:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        return loss.detach()
    

    def train(self, dataloader, val_loader, loss_f, n_epochs,dataset):
        #training of the model

        train_loss_total = np.zeros(n_epochs)
        val_loss_total = np.zeros(n_epochs)
        for epoch in range(n_epochs):
            loss=0
            for i, data in enumerate(dataloader):
                loss= loss+self.batch_step(data, loss_f)
            loss=loss/(len(dataloader))
            print("Train Loss: ",loss.item())
            train_loss_total[epoch]=loss.item()
            
            val_loss=0
            for i, data in enumerate(val_loader):
                val_loss= val_loss+self.batch_step(data, loss_f, train=False)
            val_loss = val_loss/len(val_loader)
            print("Val Loss: ",val_loss.item())
            val_loss_total[epoch]=val_loss.item()
        
        np.savetxt("/home/aih/michael.deutges/code/output/"+dataset+"_trainLoss_valLoss.csv",[train_loss_total,val_loss_total],delimiter=',')
        utils.plot_loss(train_loss_total[1:],val_loss_total,dataset)

    