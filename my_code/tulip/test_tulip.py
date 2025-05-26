import torch
from PIL import Image
import open_clip
import time

model, _, preprocess = open_clip.create_model_and_transforms('TULIP-B-16-224', pretrained='/home/linhai/code/IsaacLab/my_code/tulip/tulip-B-16-224.ckpt')
model.eval()

start_time=time.time()
image = preprocess(Image.open("/home/linhai/图片/10753_crt_V9wc8.jpg")).unsqueeze(0).to('cuda')
tokenizer = open_clip.get_tokenizer('TULIP-B-16-224')
text = tokenizer(["a girl", "a dog", "a bird"]).to('cuda')

with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    #text_features = model.encode_text(text)
    end_time=time.time()
    print(end_time-start_time)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probabilities:", similarities)