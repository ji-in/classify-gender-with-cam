import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax, interpolate
from torchvision.io.image import read_image
from torchvision.models import resnet18
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import SmoothGradCAMpp, LayerCAM, CAM
from torchcam.utils import overlay_mask

import model

net = model.VGG19()
net.load_state_dict(torch.load('gender_classifier_params.pt'))
net.eval()

img_path = 'yoona.jpg'

cam_extractor = SmoothGradCAMpp(net) # SmoothGradCAMpp은 methods/gradient.py에 존재한다.
# cam_extractor = CAM(model)
# # Get your input
# img = read_image(img_path)
# # Preprocess it for your chosen model
# input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# # Preprocess your data and feed it to the model
# out = model(input_tensor.unsqueeze(0))
# # Retrieve the CAM by passing the class index and the model output
# cams = cam_extractor(out.squeeze(0).argmax().item(), out)

# # Notice that there is one CAM per target layer (here only 1)
# for cam in cams:
#   print(cam.shape)

# # The raw CAM
# for name, cam in zip(cam_extractor.target_names, cams):
# #   plt.imshow(cam.numpy()); plt.axis('off'); plt.title(name); plt.show()
#   plt.imsave('raw_cam.png', cam.numpy()); plt.axis('off'); plt.title(name); plt.show()

# # Overlayed on the image
# for name, cam i n zip(cam_extractor.target_names, cams):
#   result = overlay_mask(to_pil_image(img), to_pil_image(cam, mode='F'), alpha=0.5)
#   plt.imsave('overlayed.png', result); plt.axis('off'); plt.title(name); plt.show()

# # Once you're finished, clear the hooks on your model
# cam_extractor.clear_hooks()

