import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImageProcessor:
    """
    A class for processing and augmenting images

    :tuple size: Target size for resizing images (width, height)
    :ImageDataGenerator augmentation_gen: Data generator for applying augmentations
    """

    def __init__(self, size=(128, 128)):
        """
        Initializes the ImageProcessor with the specified image size and
        a predefined set of augmentations

        :tuple size: Target size for image resizing, default is (128, 128)
        """

        self.size = size

        self.augmentation_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )


    def augment_image(self, image):
        """
        Augments a single image using the ImageDataGenerator

        :numpy.ndarray image: The input image to augment

        :returns numpy.ndarray: A single augmented image
        """

        for x_aug in self.augmentation_gen.flow(image, batch_size=1):

            return x_aug[0]


    def process_images(self, path:str, augment:bool = False):
        """
        Processes and optionally augments images from a given directory

        :str path: Directory path containing image files
        :bool augment: If True, generate augmented images for each input image

        :returns list: A list of processed (and optionally augmented) images as numpy arrays
        """

        images = []

        for img_file in os.listdir(path):

            image_path = os.path.join(path, img_file)
            image = cv2.imread(image_path)

            if image is not None:

                image = cv2.resize(image, self.size)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255.0

                images.append(image)

                if augment:
                    image = image.reshape((1,) + image.shape)
                    num_aug_images = 2

                    for _ in range(num_aug_images):

                        augmented_image = self.augment_image(image)
                        images.append(augmented_image)

            else:
                print(f"Warning: Unable to load image {img_file}")

        return images
