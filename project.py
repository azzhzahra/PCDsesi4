import imageio as imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def load_image(path):
    """Load image and convert to grayscale if colored"""
    img = imageio.imread(path)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2).astype(np.uint8)
    return img

def show_images(images, titles):
    """Display multiple images side by side"""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def contrast_stretching(img):
    """Apply contrast stretching"""
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = np.interp(img, (p2, p98), (0, 255))
    return img_rescale.astype(np.uint8)

def histogram_equalization(img):
    """Apply histogram equalization"""
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[img]

def gaussian_smoothing(img, sigma=1):
    """Apply Gaussian smoothing"""
    return ndimage.gaussian_filter(img, sigma=sigma)

def sharpen_image(img):
    """Apply image sharpening"""
    blur = ndimage.gaussian_filter(img, 3)
    return img + ((img - blur) * 1.5)

def main():
    # Load image
    try:
        image_path = 'gambar.png'  # Replace with your image path
        original = load_image(image_path)

        # Apply different enhancement techniques
        contrast_stretched = contrast_stretching(original)
        hist_eq = histogram_equalization(original)
        smoothed = gaussian_smoothing(original)
        sharpened = sharpen_image(original)

        # Display results
        images = [original, contrast_stretched, hist_eq, smoothed, sharpened]
        titles = ['Original', 'Contrast Stretched', 'Histogram Equalization',
        'Gaussian Smoothing', 'Sharpened']
        show_images(images, titles)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()