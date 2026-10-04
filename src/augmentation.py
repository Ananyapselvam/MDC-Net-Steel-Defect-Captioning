import albumentations as A


def get_train_transforms():

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        )
    )


def get_val_transforms():

    return A.Compose(
        [],
        #bbox_params=A.BboxParams(
          #  format="pascal_voc",
         #   label_fields=["labels"]
        #)
    )