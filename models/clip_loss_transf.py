import numpy as np
from PIL import Image
import clip
import torch
from torch import nn
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast
import os

class CLIPbase(nn.Module):
    def __init__(self):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16", framework="pt")
        self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16", framework="pt")

    def batch_scores(self, image_encodings, text_encodings):
        output = self.model(pixel_values=image_encodings['pixel_values'],
                            input_ids=text_encodings['input_ids'],
                            attention_mask=text_encodings['attention_mask']
                            )

        return output.logits_per_image

    def score(self, image_encoding, text_encoding):
        return torch.nn.functional.cosine_similarity(image_encoding, text_encoding, dim=0)

    def process_image_batch(self, images):
        processed_images = self.image_processor(images, do_rescale=False)
        return {'pixel_values': torch.from_numpy(np.array(processed_images.pixel_values)).to(self.device)}

    def process_text_batch(self, text):
        tokens = self.tokenizer(text)

        input_ids = torch.Tensor(tokens.input_ids).to(torch.int64).to(self.device)
        attention_mask = torch.Tensor(tokens.attention_mask).to(self.device)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

class CLIPLossTxt(CLIPbase):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target, norm_factor=100):
        # pred_image = pred.to(self.device)

        assert pred.shape[0] == len(target)
        n = pred.shape[0]
        
        # Normalise from [-1, 1] to [0, 1]
        pred = (pred+1) /2

        images = self.process_image_batch(pred)
        texts = self.process_text_batch(target)

        text_features = self.model.get_text_features(input_ids=texts['input_ids'],
                            attention_mask=texts['attention_mask'])
        image_features = self.model.get_image_features(pixel_values=images['pixel_values'])

        # scores = self.batch_scores(images, texts)
        sum = 0
        # for i in range(n): sum -= scores[i][i] 
        scores = torch.nn.functional.cosine_similarity(image_features, text_features, dim=1)  
        score = torch.mean(scores)

        # for i in range(n): sum -= self.score([i], [i]) 

        return -score

class CLIPLossImgTxt(CLIPbase):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target_image, target_text, norm_factor=10):
        n = pred.shape[0]
        assert n == target_image.shape[0] and n == len(target_text)

        # Normalise from [-1, 1] to [0, 1]
        pred = (pred+1) /2
        target_image = (target_image+1) /2

        fake_images = self.process_image_batch(pred)
        real_images = self.process_image_batch(target_image)

        texts = self.process_text_batch(target_text)

        real_scores = self.batch_scores(real_images, texts)
        fake_scores = self.batch_scores(fake_images, texts)
        # fake_scores = scores[:mid]
        # real_scores = scores[mid:]

        scores = torch.abs(fake_scores - real_scores)

        sum = 0
        for i in range(n): sum += scores[i][i] 

        return sum / n / norm_factor

    # def forward(self, pred, target_image, target_text, norm_factor=10):
    #     n = pred.shape[0]
    #     assert n == target_image.shape[0] and n == len(target_text)

    #     all_images = torch.cat([pred, target_image])
    #     all_text = target_text + target_text

    #     # Normalise from [-1, 1] to [0, 1]
    #     all_images = (all_images+1) /2

    #     images = self.process_image_batch(all_images)
    #     texts = self.process_text_batch(all_text)

    #     scores = self.batch_scores(images, texts)
    #     # fake_scores = scores[:mid]
    #     # real_scores = scores[mid:]

    #     # scores = torch.abs(fake_scores - real_scores)

    #     sum = 0
    #     for i in range(n): sum += torch.abs(scores[i][i] - scores[i + n][i + n]) 

    #     return sum / n / norm_factor

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