import numpy as np
from PIL import Image
import clip
import torch
from torch import nn

class CLIPbase(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)

    def score(self, image, text):
        image = self.preprocess(tensor_to_image(image)).unsqueeze(0).to(self.device)
        text = clip.tokenize([text]).to(self.device)

        img_features = self.model.encode_image(image)
        txt_features = self.model.encode_text(text)
        # return 0
        return torch.nn.functional.cosine_similarity(img_features, txt_features, dim=1)

class CLIPLossTxt(CLIPbase):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target):
        # pred_image = pred.to(self.device)

        assert pred.shape[0] == target.shape[0]
        n = pred.shape[0]
        tot_loss = None

        for i, img in enumerate(pred):
            score = self.score(img, target[i])
            tot_loss = score if tot_loss == None else tot_loss + score

        return tot_loss / n

class CLIPLossImgTxt(CLIPbase):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target_image, target_text):
        n = pred.shape[0]
        assert n == target_image.shape[0] and n == target_text.shape[0]

        tot_loss = None
        for i, img in enumerate(pred):
            txt = target_text[i]
            target_img = target_image[i]

            score_pred = self.score(img, txt)
            score_target = self.score(target_img, txt)
            loss = torch.abs(score_pred - score_target)

            tot_loss = loss if tot_loss == None else tot_loss + loss

        return tot_loss / n

class CLIPLossImg(CLIPbase):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        #TODO: Implement
        raise NotImplementedError()


#### UTILS ####         #TODO move
def tensor_to_image(tensor: torch.Tensor):
    # print(tensor.shape)
    tensor = tensor.cpu().detach()
    tensor = tensor.permute(2, 1, 0)
    tensor = np.array(tensor*255, dtype=np.uint8)
    # print(tensor.shape)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def batch_tensor_to_images(tensor: torch.Tensor):
    # print(tensor.shape)
    tensor = tensor.cpu().detach()
    tensor = tensor.permute(0, 3, 2, 1)
    tensor = np.array(tensor*255, dtype=np.uint8)
    # print(tensor.shape)
    if np.ndim(tensor)>3:
        images = [Image.fromarray(img) for img in tensor]
        return images
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)