#Training on one and testing on three datasets

#IMPORTS
import yaml
config_path = "config.yaml"
with open(config_path) as file:
  config = yaml.safe_load(file)
import torch
import numpy as np
import src.utils.utils as utils
import torch.utils.data as data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit,StratifiedKFold
import src.datasets.Dataset as Dataset 
from src.models.NCA import MaxNCA, ConvNCA, SimpleNCA
from src.losses.LossFunctions import BCELoss
from src.agents.Agent import Agent

#CONFIGURATION OF EXPERIMENT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#DATALOADING
AML_data_path = "/datasets/Matek19/"
PBC_data_path = "/datasets/Acevedo20/"
MLL_data_path = "/datasets/INT20/"

X_AML,y_AML = utils.get_data_AML(AML_data_path,show_distribution=False)
X_PBC,y_PBC = utils.get_data_PBC(PBC_data_path,show_distribution=False)
X_MLL,y_MLL = utils.get_data_MLL(MLL_data_path,show_distribution=False)
X_AML=np.asarray(X_AML)
y_AML=np.asarray(y_AML)
X_PBC=np.asarray(X_PBC)
y_PBC=np.asarray(y_PBC)
X_MLL=np.asarray(X_MLL)
y_MLL=np.asarray(y_MLL)
fold=config["fold"]
skf_AML = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
skf_AML.get_n_splits(X_AML, y_AML)
for i,(train_index, test_index) in enumerate(skf_AML.split(X_AML, y_AML)):
    if i != fold:
        continue
    X_AML_train, X_AML_test = X_AML[train_index], X_AML[test_index]
    y_AML_train, y_AML_test = y_AML[train_index], y_AML[test_index]

skf_PBC = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
skf_PBC.get_n_splits(X_PBC, y_PBC)
for i,(train_index, test_index) in enumerate(skf_PBC.split(X_PBC, y_PBC)):
    if i != fold:
        continue
    X_PBC_train, X_PBC_test = X_PBC[train_index], X_PBC[test_index]
    y_PBC_train, y_PBC_test = y_PBC[train_index], y_PBC[test_index]

skf_MLL = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
skf_MLL.get_n_splits(X_MLL, y_MLL)
for i,(train_index, test_index) in enumerate(skf_MLL.split(X_MLL, y_MLL)):
    if i != fold:
        continue
    X_MLL_train, X_MLL_test = X_MLL[train_index], X_MLL[test_index]
    y_MLL_train, y_MLL_test = y_MLL[train_index], y_MLL[test_index]


#X_AML_train, X_AML_test, y_AML_train, y_AML_test = train_test_split(X_AML, y_AML, test_size=0.20, random_state=2)
#X_PBC_train, X_PBC_test, y_PBC_train, y_PBC_test = train_test_split(X_PBC, y_PBC, test_size=0.20, random_state=2)
#X_MLL_train, X_MLL_test, y_MLL_train, y_MLL_test = train_test_split(X_MLL, y_MLL, test_size=0.20, random_state=2)

AML_train_dataset = Dataset.WBC_Dataset(X_AML_train,y_AML_train, augment=True, resize=config["resize"],dataset="AML")
AML_val_dataset = Dataset.WBC_Dataset(X_AML_test,y_AML_test, resize=config["resize"],dataset="AML")
PBC_train_dataset = Dataset.WBC_Dataset(X_PBC_train,y_PBC_train, augment=True, resize=config["resize"],dataset="PBC")
PBC_val_dataset = Dataset.WBC_Dataset(X_PBC_test,y_PBC_test, resize=config["resize"],dataset="PBC")
MLL_train_dataset = Dataset.WBC_Dataset(X_MLL_train,y_MLL_train, augment=True, resize=config["resize"],dataset="MLL")
MLL_val_dataset = Dataset.WBC_Dataset(X_MLL_test,y_MLL_test, resize=config["resize"],dataset="MLL")

if config["balance"]:
    AML_sampler = data.WeightedRandomSampler(weights=utils.get_weights(y_AML_train), num_samples=len(AML_train_dataset), replacement=True)
    PBC_sampler = data.WeightedRandomSampler(weights=utils.get_weights(y_PBC_train), num_samples=len(PBC_train_dataset), replacement=True)
    MLL_sampler = data.WeightedRandomSampler(weights=utils.get_weights(y_MLL_train), num_samples=len(MLL_train_dataset), replacement=True)
else:
    sampler=None

AML_train_loader = data.DataLoader(AML_train_dataset,sampler=AML_sampler,batch_size=config["batch_size"])
AML_val_loader = data.DataLoader(AML_val_dataset, batch_size=1)
PBC_train_loader = data.DataLoader(PBC_train_dataset,sampler=PBC_sampler,batch_size=config["batch_size"])
PBC_val_loader = data.DataLoader(PBC_val_dataset, batch_size=1)
MLL_train_loader = data.DataLoader(MLL_train_dataset,sampler=MLL_sampler,batch_size=config["batch_size"])
MLL_val_loader = data.DataLoader(MLL_val_dataset, batch_size=1)

#TRAINING
if config["model"]=="MaxNCA":
    model=MaxNCA(channel_n=config["channel_n"], hidden_size=config["hidden_size"])
elif config["model"]=="ConvNCA":
    model=ConvNCA(channel_n=config["channel_n"], hidden_size=config["hidden_size"])
else:
    model=SimpleNCA(channel_n=config["channel_n"], hidden_size=config["hidden_size"])

model.to(device)
agent=Agent(model,config["steps"],config["channel_n"],config["batch_size"])
if config["train_set"]=="AML":
    agent.train(AML_train_loader,AML_val_loader,BCELoss,config["n_epochs"],config["name"]+"AML")
    utils.get_confusion_matrix(model,agent,AML_val_loader,config["name"]+"_"+config["train_set"]+"_AML",config["steps"])
elif config["train_set"]=="PBC":
    agent.train(PBC_train_loader,PBC_val_loader,BCELoss,config["n_epochs"],config["name"]+"PBC")
    utils.get_confusion_matrix(model,agent,PBC_val_loader,config["name"]+"_"+config["train_set"]+"_PBC",config["steps"])
elif config["train_set"]=="MLL":
    agent.train(MLL_train_loader,MLL_val_loader,BCELoss,config["n_epochs"],config["name"]+"MLL")
    utils.get_confusion_matrix(model,agent,MLL_val_loader,config["name"]+"_"+config["train_set"]+"_MLL",config["steps"])
torch.save(model.state_dict(), "/models/"+config["name"]+"_trained_on_"+config["train_set"])


#RESULTS AND VISUALIZATION
utils.get_confusion_matrix(model,agent,AML_val_loader,config["name"]+"_"+config["train_set"]+"_AML",config["steps"])
utils.get_confusion_matrix(model,agent,PBC_val_loader,config["name"]+"_"+config["train_set"]+"_PBC",config["steps"])
utils.get_confusion_matrix(model,agent,MLL_val_loader,config["name"]+"_"+config["train_set"]+"_MLL",config["steps"])