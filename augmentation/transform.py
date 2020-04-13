import numpy as np
import augmentation.transform_function as transform_func
from augmentation.transform_function import DETransform


class AugmentName:

    IDENTITY = DETransform(transform_func=transform_func.identity, transform_range=np.linspace(0, 0, 10))
    TRANSLATE_X = DETransform(transform_func=transform_func.translate_x, transform_range=np.linspace(0, 0.2, 10))
    TRANSLATE_Y = DETransform(transform_func=transform_func.translate_y, transform_range=np.linspace(0, 0.2, 10))
    SCALE_WIDTH = DETransform(transform_func=transform_func.scale_width, transform_range=np.linspace(0, 0.1, 10))
    SCALE_HEIGHT = DETransform(transform_func=transform_func.scale_height, transform_range=np.linspace(0, 0.1, 10))
    ROTATE = DETransform(transform_func=transform_func.rotate, transform_range=np.linspace(0, 60, 10))
    SHEAR_X = DETransform(transform_func=transform_func.shear_x, transform_range=np.linspace(0, 0.3, 10))
    SHEAR_Y = DETransform(transform_func=transform_func.shear_y, transform_range=np.linspace(0, 0.3, 10))
    COLOR = DETransform(transform_func=transform_func.color, transform_range=np.linspace(0.0, 0.9, 10))
    POSTERIZE = DETransform(transform_func=transform_func.posterize, transform_range=np.round(np.linspace(8, 4, 10), 0).astype(np.int))
    SOLARIZE = DETransform(transform_func=transform_func.solarize, transform_range=np.linspace(256, 231, 10))
    CONTRAST = DETransform(transform_func=transform_func.contrast, transform_range=np.linspace(0.0, 0.5, 10))
    SHARPNESS = DETransform(transform_func=transform_func.sharpness, transform_range=np.linspace(0.0, 0.9, 10))
    BRIGHTNESS = DETransform(transform_func=transform_func.brightness, transform_range=np.linspace(0.0, 0.3, 10))
    AUTOCONSTRAST = DETransform(transform_func=transform_func.autocontrast, transform_range=np.linspace(0, 0, 10))
    INVERT = DETransform(transform_func=transform_func.invert, transform_range=np.linspace(0, 0, 10))


class DERandAugment:
    TRANSFORMS = [
            AugmentName.IDENTITY,
            AugmentName.TRANSLATE_X,
            AugmentName.TRANSLATE_Y,
            AugmentName.SCALE_WIDTH,
            AugmentName.SCALE_HEIGHT,
            AugmentName.ROTATE,
            AugmentName.SHEAR_X,
            AugmentName.SHEAR_Y,
            AugmentName.COLOR,
            AugmentName.POSTERIZE,
            AugmentName.SOLARIZE,
            AugmentName.CONTRAST,
            AugmentName.SHARPNESS,
            AugmentName.BRIGHTNESS,
            AugmentName.AUTOCONSTRAST,
            AugmentName.INVERT,
    ]

    def __init__(self, n_apply_transform=5, magnitude=6):
        self.n_apply_transform = n_apply_transform
        self.magnitude = magnitude

        return

    def __call__(self, images, annotations=None):
        if annotations is None:
            annotations = []

        assert len(images) == len(annotations) or len(annotations) == 0, "Number of anotation must be zero or equal to number of image"

        result = []
        for i in range(len(images)):
            annotation = None
            if len(annotations) > 0:
                annotation = annotations[i]

            image, annotation = self.transform(
                image=images[i],
                annotations=annotation
            )

            result.append({"image": image, "annotations": annotation})

        return result

    def transform(self, image, annotations=None):
        ops = np.random.choice(DERandAugment.TRANSFORMS, self.n_apply_transform)
        ops = [ops[i].transform_func(image, ops[i].transform_range[np.random.randint(low=0, high=self.magnitude)]) for i in range(len(ops))]

        for i in range(len(ops)):
            image, annotations = ops[i](image, annotations)

        return image, annotations

