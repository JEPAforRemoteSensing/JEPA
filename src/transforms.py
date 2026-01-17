from torchvision.transforms import v2

def make_transforms(
    num_channels,
    crop_size=224,
    crop_scale=(0.8, 1.0),
    horizontal_flip=True,
    vertical_flip=True,
    gaussian_blur=True,
):
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
            mean=[-17.453, -11.222, 436.825, 684.856, 653.592, 1113.942, 2371.036, 2848.349, 2978.488, 2317.410, 1453.706, 3101.315],
            std=[3.781, 3.495, 237.397, 276.302, 432.002, 388.312, 547.779, 726.583, 795.423, 645.402, 704.461, 762.472]
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
