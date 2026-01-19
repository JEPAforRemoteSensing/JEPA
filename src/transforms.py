from torchvision.transforms import v2

def make_transforms(
    num_channels,
    crop_size=224,
    crop_scale=(0.8, 1.0),
    horizontal_flip=True,
    vertical_flip=True,
    gaussian_blur=True,
):
    s1_mean = [0.5786, 0.5114]
    s1_std  = [0.097, 0.0951]

    s2_mean = [0.1248, 0.1712, 0.152, 0.259, 0.4742, 0.4747, 0.4513, 0.4291, 0.2795, 0.4923]
    s2_std  = [0.0673, 0.0686, 0.1001, 0.09, 0.1094, 0.121, 0.1205, 0.1192, 0.1351, 0.121]
    transform = v2.Compose([
        v2.RandomResizedCrop(crop_size, scale=crop_scale),
        v2.RandomHorizontalFlip() if horizontal_flip else v2.Identity(),
        v2.RandomVerticalFlip() if vertical_flip else v2.Identity(),
        v2.RandomChoice([
            v2.RandomRotation((0, 0)),
            v2.RandomRotation((90, 90)),
            v2.RandomRotation((180, 180)),
            v2.RandomRotation((270, 270))
        ]),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))],
            p=0.5) if gaussian_blur else v2.Identity(),
        v2.Normalize(
            mean=s1_mean+s2_mean,
            std=s1_std+s2_std
        )
    ])

    return transform

def make_transforms_rgb(
    num_channels,
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=True,
    vertical_flip=True,
    color_distortion=True,
    gaussian_blur=True,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    transform = v2.Compose([
        v2.RandomResizedCrop(crop_size, scale=crop_scale),
        v2.RandomHorizontalFlip() if horizontal_flip else v2.Identity(),
        v2.RandomVerticalFlip() if vertical_flip else v2.Identity(),
        v2.RandomRotation(),
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
