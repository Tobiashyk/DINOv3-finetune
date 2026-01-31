import glob
from PIL import Image

def load_images(path):
    samples = glob.glob(f"{path}/*.png")
    samples.sort()  # 确保文件加载顺序是确定的
    images = [Image.open(s).convert('RGB') for s in samples]
    return images