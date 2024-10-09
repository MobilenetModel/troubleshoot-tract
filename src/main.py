import typing

import numpy as np
from tensorflow.keras.applications import MobileNetV3Small  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore

# Load pre-trained MobileNetV3Small model.
MODEL = MobileNetV3Small(
    weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
)


def preprocess_image(
    path_to_image: str,
) -> np.ndarray[typing.Any, typing.Any]:
    img224x224 = image.load_img(path_to_image, target_size=(224, 224))
    # Save image to disk under a /pp_images directory.
    # img224x224.save(f"pp_images/{os.path.basename(path_to_image)}")
    img_array = image.img_to_array(img224x224)
    np_expand_dims_img_array = np.expand_dims(img_array, axis=0)
    return np_expand_dims_img_array


TEST_IMAGE_PATH = "test_image.jpg"
model_prediction = MODEL.predict(preprocess_image(TEST_IMAGE_PATH))
print(model_prediction)
