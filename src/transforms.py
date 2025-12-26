from torchvision.transforms import v2

def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    
    transform = v2.Compose([
        v2.RandomResizedCrop(crop_size, scale=crop_scale),
        v2.RandomHorizontalFlip() if horizontal_flip else v2.Identity(),
        v2.RandomApply(
            [v2.ColorJitter(0.8*color_jitter, 0.8*color_jitter, 0.8*color_jitter, 0.2*color_jitter)],
            p=0.8) if color_distortion else v2.Identity(),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))],
            p=0.5) if gaussian_blur else v2.Identity(),
        v2.ToTensor(),
        v2.Normalize(mean=normalization[0], std=normalization[1])
    ])

    return transform
