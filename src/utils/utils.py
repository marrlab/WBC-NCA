import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torchvision.transforms import v2
from PIL import Image

def get_data(data_path,show_distribution=True,dataset="AML"):
    #returns list of image paths and list of corresponding labels

    image_paths = []
    labels = []

    for dirs in os.listdir(data_path):
        folder_path = os.path.join(data_path, dirs)
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.tiff') or file.endswith('.TIF'):
                image_path = os.path.join(folder_path, file)
                image_paths.append(image_path)
                    
                if "basophil" == dirs:
                    label = 0
                elif "eosinophil" == dirs:
                    label = 1
                elif "erythroblast" == dirs:
                    label = 2
                elif "lymphocyte_typical" == dirs:
                    label = 3
                elif "metamyelocyte" == dirs:
                    label = 4
                elif "monocyte" == dirs:
                    label = 5
                elif "myeloblast" == dirs:
                    label = 6
                elif "myelocyte" == dirs:
                    label = 7
                elif "neutrophil_band" == dirs:
                    label = 8
                elif "neutrophil_segmented" == dirs:
                    label = 9
                elif "promyelocyte" == dirs:
                    label = 10
                labels.append(label)
    
    if show_distribution==True:
        x = ["basophil","eosinophil","erythroblast","lymphocyte_typical","metamyelocyte","monocyte","myeloblast","myelocyte","neutrophil_band","neutrophil_segmented","promyelocyte"]
        y=[Counter(labels)[i] for i in range(11)]
        fig=plt.figure()
        plt.xticks(rotation=45)
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.bar(x, y)
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.savefig("/home/aih/michael.deutges/code/output/data_distribution_"+dataset+".png")

    return image_paths, labels

def get_weights(y):
    #get weights for the data balancing
    class_counts = Counter(y)
    class_counts = np.array([class_counts[i] for i in range(15)])
    class_weights = 1/(class_counts+0.001)
    sample_weights = [class_weights[i] for i in y]
    return sample_weights

def get_confusion_matrix(model,agent,test_loader,dataset,steps):
    #computes confusion matrix
    output="home/aih/michael.deutges/code/output/"
    confusion_matrix=np.zeros([13,13])
    iterable = iter(test_loader)
    for i in enumerate(test_loader):
        x,s=next(iterable)
        out,_=model(agent.make_seed(x), steps=steps,fire_rate=0.5)
        out=out.detach()
        pred=out[0]
        sig=torch.nn.Sigmoid()
        pred=sig(pred).cpu().numpy()
        pred=np.argmax(pred)
        s=s.cpu()
        label=np.argmax(s[0])
        confusion_matrix[label,pred] += 1
    np.savetxt("/home/aih/michael.deutges/code/output/confusion_matrix_"+dataset,confusion_matrix.astype(int),fmt="%d",delimiter=',')
    
def get_dice_scores(model,agent,test_loader,dataset,steps):
    #computes dice scores for each image
    output="home/aih/michael.deutges/code/output/"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dice_scores=[]
    iterable = iter(test_loader)
    j=-1
    for i in enumerate(test_loader):
        j=j+1
        x,s=next(iterable)
        out,_=model(agent.make_seed(x), steps=steps,fire_rate=0.5)
        out=out.detach()
        input=out
        target=s
        input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target).to(device)
        intersection = (input * target).sum()
        dice = (2.*intersection)/(input.sum() + target.sum())
        dice_scores.append(dice.cpu().item())
        
    print(dice_scores)
    np.savetxt("/home/aih/michael.deutges/code/output/dice_scores_"+dataset,dice_scores,fmt="%d",delimiter=',')
    print(dice_scores.mean())

def animate_activation(model,agent,val_loader,steps):
    #visualizes channel activations over time
    import matplotlib.animation as animation
    from IPython.display import HTML
    output="home/aih/michael.deutges/code/output/"
    names=["BAS","EBO","EOS","KSC","LYA","LYT","MMZ","MOB","MON","MYB","MYO","NGB","NGS","PMB","PMO"]
    x,s=next(iter(val_loader))
    plt.rcParams["figure.figsize"] = (64,64)
    plt.figure(figsize=(64,64))
    fig, ax = plt.subplots()
    ims = []
    for i in range(steps+1):
        pred,feat_map=model(agent.make_seed(x), steps=i-1,fire_rate=0.5)
        pred=pred.detach()
        feat_map=feat_map.detach()
        sig=torch.nn.Sigmoid()
        feat_map=sig(feat_map)
        feat_map=feat_map[0].cpu()
        feat_map[:,:,0]=0*feat_map[:,:,0]
        feat_map[:,:,1]=0*feat_map[:,:,0]
        feat_map[:,:,2]=0*feat_map[:,:,0]
        #feat_map = torch.reshape(feat_map.permute(0,2,1),(feat_map.shape[0]*8,feat_map.shape[1]*8))#feat_map.shape[2]))
        feat_map=torch.cat([torch.cat([feat_map[:,:,i+8*j] for i in range(8)],axis=1) for j in range(8)],axis=0)
        #feat_map=feat_map.numpy()
        plt.gray()
        im = ax.imshow(feat_map, animated=True)
        if i == 0:
            ax.imshow(feat_map)  # show an initial one first
        ims.append([im])


    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                    repeat_delay=10000)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('/home/aih/michael.deutges/code/output/channel_visualization.gif', writer=writer)
    return HTML(ani.to_jshtml())

def visualize_activations(model,agent,steps,sample,name):
    #plots channels of the NCA
    sample=sample[None,:,:,:]

    plt.figure("visualise", (64,64))
    out,feature_map=model(agent.make_seed(sample), steps=steps,fire_rate=0.5)
    feature_map=feature_map.detach()
    #sig=torch.nn.Sigmoid()
    #feature_map=sig(feature_map)
    feature_map=feature_map.cpu()
    feature_map[0,:,:,0]=0*feature_map[0,:,:,0]
    feature_map[0,:,:,1]=0*feature_map[0,:,:,0]
    feature_map[0,:,:,2]=0*feature_map[0,:,:,0]
    for i in range(8):
        for j in range(8):
            plt.subplot(8,8,8*i+j+1)
            plt.gray()
            plt.imshow(feature_map[0,:,:,8*i+j])
    plt.savefig("/home/aih/michael.deutges/code/test_samples/channel_activation"+name+".png")
    return

def get_test_samples():
    #gets sample images for tests
    imgs_path=["/home/aih/michael.deutges/code/test_samples/AML_LYT.tiff",
            "/home/aih/michael.deutges/code/test_samples/AML_MON.tiff",
            "/home/aih/michael.deutges/code/test_samples/AML_NGS.tiff",
            "/home/aih/michael.deutges/code/test_samples/PBC_LYT.jpg",
            "/home/aih/michael.deutges/code/test_samples/PBC_MON.jpg",
            "/home/aih/michael.deutges/code/test_samples/PBC_NGS.jpg",
            "/home/aih/michael.deutges/code/test_samples/MLL_LYT.TIF",
            "/home/aih/michael.deutges/code/test_samples/MLL_MON.TIF",
            "/home/aih/michael.deutges/code/test_samples/MLL_NGS.TIF"]
    norm = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.80382601, 0.70016377, 0.83305684],std=[0.1863151, 0.28160227, 0.10350788])])
    img = Image.open(imgs_path[0])
    img = img.resize((64,64))
    img1 = norm(img).permute(1,2,0)
    img = Image.open(imgs_path[1])
    img = img.resize((64,64))
    img2 = norm(img).permute(1,2,0)
    img = Image.open(imgs_path[2])
    img = img.resize((64,64))
    img3 = norm(img).permute(1,2,0)
    norm = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.82479313, 0.69093741, 0.70071333],std=[0.20754552, 0.23684978, 0.09308279])])
    img = Image.open(imgs_path[3])
    img = img.resize((64,64))
    img4 = norm(img).permute(1,2,0)
    img = Image.open(imgs_path[4])
    img = img.resize((64,64))
    img5 = norm(img).permute(1,2,0)
    img = Image.open(imgs_path[5])
    img = img.resize((64,64))
    img6 = norm(img).permute(1,2,0)
    norm = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.74082622, 0.65207104, 0.77892128],std=[0.20470387, 0.26758623, 0.17912261])])
    img = Image.open(imgs_path[6])
    img = img.resize((64,64))
    img7 = norm(img).permute(1,2,0)
    img = Image.open(imgs_path[7])
    img = img.resize((64,64))
    img8 = norm(img).permute(1,2,0)
    img = Image.open(imgs_path[8])
    img = img.resize((64,64))
    img9 = norm(img).permute(1,2,0)

    return img1,img2,img3,img4,img5,img6,img7,img8,img9

def plot_loss(train,val,dataset):
    #plots train and validation loss curve
    output="home/aih/michael.deutges/code/output/"
    fig=plt.figure()
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.plot(train, label="Train")
    plt.plot(val, label="Val")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("/home/aih/michael.deutges/code/output/loss_plot_"+dataset+".png")
    return

def get_data_AML(data_path,show_distribution=True):
    #returns list of image paths and list of corresponding labels

    image_paths = []
    labels = []

    for dirs in os.listdir(data_path):
        folder_path = os.path.join(data_path, dirs)
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.tiff'):
                image_path = os.path.join(folder_path, file)
                
                if "BAS" in file:
                    label = 0
                elif "EBO" in file:
                    label = 2
                elif "EOS" in file:
                    label = 1
                elif "KSC" in file:
                    label = 12
                elif "LYA" in file:
                    label = 11
                elif "LYT" in file:
                    label = 10
                elif "MMZ" in file:
                    label = 6
                elif "MOB" in file:
                    label = 9
                elif "MON" in file:
                    label = 9
                elif "MYB" in file:
                    label = 5
                elif "MYO" in file:
                    label = 3
                elif "NGB" in file:
                    label = 7
                elif "NGS" in file:
                    label = 8
                elif "PMB" in file:
                    continue
                    label = 13
                elif "PMO" in file:
                    label = 4
                labels.append(label)
                image_paths.append(image_path)
    
    if show_distribution==True:
        x = ['basophil','eosinophil','erythroblast','myeloblast','promyelocyte','myelocyte','metamyelocyte','neutrophil_banded','neutrophil_segmented','monocyte','lymphocyte_typical','lymphocyte_atypical','smudge_cell']
        y=[Counter(labels)[i] for i in range(13)]
        fig=plt.figure()
        plt.xticks(rotation=45)
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.bar(x, y)
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.savefig("/home/aih/michael.deutges/code/output/data_distribution_AML.png")

    return image_paths, labels

def get_data_PBC(data_path,show_distribution=True):
    #returns list of image paths and list of corresponding labels

    image_paths = []
    labels = []

    for dirs in os.listdir(data_path):
        if ".DS_" in dirs:
            continue
        folder_path = os.path.join(data_path, dirs)
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.tiff'):
                image_path = os.path.join(folder_path, file)
                    
                if "basophil" in dirs:
                    label = 0
                elif "eosinophil" in dirs:
                    label = 1
                elif "erythroblast" in dirs:
                    label = 2
                elif "IG" in dirs:
                    continue
                    label = 13
                elif "PMY" in dirs:
                    label = 4
                elif "MY" in dirs:
                    label = 5
                    if "MMY" in dirs:
                        label = 6
                elif "lymphocyte" in dirs:
                    label = 10
                elif "monocyte" in dirs:
                    label = 9
                elif "NEUTROPHIL" in dirs:
                    continue
                    label = 13
                elif "BNE" in file:
                    label = 7
                elif "SNE" in file:
                    label = 8
                elif "platelet" in file:
                    continue
                    label = 13
                labels.append(label)
                image_paths.append(image_path)
        
    if show_distribution==True:
        x = ['basophil','eosinophil','erythroblast','myeloblast','promyelocyte','myelocyte','metamyelocyte','neutrophil_banded','neutrophil_segmented','monocyte','lymphocyte_typical','lymphocyte_atypical','smudge_cell']
        y=[Counter(labels)[i] for i in range(13)]
        fig=plt.figure()
        plt.xticks(rotation=45)
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.bar(x, y)
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.savefig("/home/aih/michael.deutges/code/output/data_distribution_PCB.png")

    return image_paths, labels

def get_data_MLL(data_path,show_distribution=True):
    #returns list of image paths and list of corresponding labels

    image_paths = []
    labels = []

    for dirs in os.listdir(data_path):
        folder_path = os.path.join(data_path, dirs)
       
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.tiff') or file.endswith('.TIF'):
                image_path = os.path.join(folder_path, file)
                    
                if "01" in dirs:
                    label = 2
                elif "04" in dirs:
                    continue
                    label = 13
                elif "05" in dirs:
                    label = 9
                elif "08" in dirs:
                    label = 11
                elif "09" in dirs:
                    label = 0
                elif "10" in dirs:
                    label = 1
                elif "11" in dirs:
                    label = 7
                elif "12" in dirs:
                    label = 11
                elif "13" in dirs:
                    label = 3
                elif "14" in dirs:
                    label = 10
                elif "15" in dirs:
                    label = 8
                elif "16" in dirs:
                    continue
                    label = 13
                elif "17" in dirs:
                    label = 12
                elif "18" in dirs:
                    label = 4
                elif "19" in dirs:
                    label = 5
                elif "20" in dirs:
                    label = 6
                elif "21" in dirs:
                    continue
                    label = 13
                elif "22" in dirs:
                    continue
                    label = 13
                labels.append(label)
                image_paths.append(image_path)
    
    if show_distribution==True:
        x = ['basophil','eosinophil','erythroblast','myeloblast','promyelocyte','myelocyte','metamyelocyte','neutrophil_banded','neutrophil_segmented','monocyte','lymphocyte_typical','lymphocyte_atypical','smudge_cell']
        y=[Counter(labels)[i] for i in range(13)]
        fig=plt.figure()
        plt.xticks(rotation=45)
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.bar(x, y)
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.savefig("/home/aih/michael.deutges/code/output/data_distribution_MLL.png")

    return image_paths, labels