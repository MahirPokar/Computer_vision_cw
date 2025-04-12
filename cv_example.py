import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 test dataset
transform = transforms.Compose([transforms.ToTensor()])
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Select an image and convert to NumPy
image_tensor, label = cifar10[9]  # You can change the index here
image_np = (image_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

# Convert to grayscale and upscale
gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
gray_upscaled = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_CUBIC)

# Apply ORB
orb = cv2.ORB_create(nfeatures=5)
keypoints, descriptors = orb.detectAndCompute(gray_upscaled, None)

# Draw keypoints
output = cv2.drawKeypoints(
    gray_upscaled, keypoints, None, color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Save and display
plt.figure(figsize=(5, 5))
plt.imshow(output, cmap='gray')
plt.title("ORB Keypoints on CIFAR-10 Image")
plt.axis('off')
plt.tight_layout()
plt.savefig("orb_keypoints_example.png", dpi=300)
plt.show()
