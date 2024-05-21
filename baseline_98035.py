from transformers import (
    AutoModel,
    CLIPProcessor,
    CLIPModel
)
import os
import pandas as pd
from PIL import Image
import torch
from PIL import ImageEnhance , Image , ImageFilter


src_dir = '/content/test/images'
query_dir = '/content/queries/queries'
submission = pd.read_csv('/content/sample_submission.csv')
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336').cuda().eval()
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14-336')

from PIL import ImageEnhance, Image , ImageFilter

def enhance_image(image_path, factor=1.5, blur_radius=0.75):
    # Open the image
    img = Image.open(image_path).convert('RGB')
    
    # Apply color enhancement (adjust the factor as needed)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(factor)

    enhancer_brightness = ImageEnhance.Brightness(img)
    img_brightness = enhancer_brightness.enhance(1.5)
    
    # Enhance contrast
    enhancer_contrast = ImageEnhance.Contrast(img_brightness)
    img = enhancer_contrast.enhance(1.5)

    # Apply Gaussian blur
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return img_blur

submission['dot_class'] = 22
submission['cosine_class'] = 22
with torch.no_grad():
    query_images = []
    query_classes = []
    for file in os.listdir(query_dir):
      # Enhance query image
      enhanced_query_image = enhance_image(os.path.join(query_dir, file))
    
      # Process the enhanced image and extract features
      inputs = processor(images=[enhanced_query_image], return_tensors='pt').to('cuda')
      outputs = model.get_image_features(inputs.pixel_values).cpu()
      outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
      query_images.append(outputs)
      query_classes.append(int(file[:-4]))

    query_images = torch.cat(query_images)

    for idx, row in submission.iterrows():
      if not pd.isna(row['class']):
          continue
    
      # Enhance source image
      enhanced_source_image = enhance_image(os.path.join(src_dir, row['img_file']))
    
      # Process the enhanced source image and extract features
      inputs = processor(images=[enhanced_source_image], return_tensors='pt').to('cuda')
      outputs = model.get_image_features(inputs.pixel_values).cpu()
      outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
    
      # Continue with the rest of the code for similarity search
      values = outputs @ query_images.T
      if values.softmax(1).max() > .055:
          submission.at[idx, 'dot_class'] = query_classes[values.argmax().numpy().tolist()]
    
      cosine = torch.cosine_similarity(outputs, query_images)
      if cosine.max() > 0.8:
          submission.at[idx, 'cosine_class'] = query_classes[cosine.argmax().numpy().tolist()]


    sub = submission[['img_file',]]
    sub['class'] = submission['dot_class']
    sub.to_csv('dot_product.csv', index=False)
    sub['class'] = submission['cosine_class']
    sub.to_csv('cosine_similarity.csv', index=False)