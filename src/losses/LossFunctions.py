#Source: Implementation from Kalkhof et al, Med-NCA (https://github.com/MECLabTUDA/M3D-NCA)

import torch

class DiceLoss(torch.nn.Module):
    r"""Dice Loss
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceLoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        if self.useSigmoid:
            input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target)
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice
    
class DiceBCELoss(torch.nn.Module):
    r"""Dice BCE Loss
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceBCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)
        
        intersection = (input * target).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)  
        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class BCELoss(torch.nn.Module):
    r"""BCE Loss
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(BCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)

        BCE = torch.nn.functional.binary_cross_entropy(input.float(), target.float(), reduction='mean')
        return BCE