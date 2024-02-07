import numpy as np
from PIL import Image
import clip
import torch
from torch import nn
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast, CLIPTokenizer
import os

class CLIPbase(nn.Module):
    def __init__(self):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16", framework="pt")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16", framework="pt")

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
        tokens = self.tokenizer(text, padding=True)

        input_ids = torch.Tensor(tokens.input_ids).to(torch.int64).to(self.device)
        attention_mask = torch.Tensor(tokens.attention_mask).to(self.device)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

class CLIPLossTxt(CLIPbase):
    def __init__(self):
        super().__init__()

    def use_whole_model(self, pred, target_text):
        assert pred.shape[0] == len(target_text)
        n = pred.shape[0]

        # Normalise from [-1, 1] to [0, 1]
        pred = (pred+1) /2

        images = self.process_image_batch(pred)
        texts = self.process_text_batch(target_text)

        scores = self.batch_scores(images, texts)
        sum = 0
        for i in range(n): sum -= scores[i][i] 

        return -sum / n /100
        

    def forward(self, pred: torch.Tensor, target, norm_factor=100):

        assert pred.shape[0] == len(target)
        
        # Normalise from [-1, 1] to [0, 1]
        pred = (pred+1) /2

        images = self.process_image_batch(pred)
        texts = self.process_text_batch(target)

        text_features = self.model.get_text_features(input_ids=texts['input_ids'],
                            attention_mask=texts['attention_mask'])
        image_features = self.model.get_image_features(pixel_values=images['pixel_values'])

        scores = torch.nn.functional.cosine_similarity(image_features, text_features, dim=1)  
        score = torch.mean(scores)

        return -score
