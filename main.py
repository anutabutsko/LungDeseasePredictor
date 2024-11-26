# internals
from lib.image_processor import ImageProcessor
from lib.cnn_classifier import CNNClassifier

# externals
import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    # path to data
    data_path = 'data'
    directories = [
        'NORMAL', 
        'PNEUMONIA', 
        'COVID', 
        'CARDIOMEGALY'
    ]

    # instantiate image processor
    image_processor = ImageProcessor(augment=True)

    # set empty lists to store image data and labels
    data = []
    labels = []

    # process images
    for label, directory in enumerate(directories):

        directory_path = {f'{data_path}/{directory}'}

        augment = True if directory in ['NORMAL', 'CARDIOMEGALY'] else False

        processed_images = image_processor.process_images(directory_path, augment)

        # save images and labels
        data.extend(processed_images)
        labels.extend([label] * len(processed_images))

    X = np.array(data)
    Y = np.array(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    input_shape = (128, 128, 3)
    num_classes = 4

    cnn_model = CNNClassifier(X_train, Y_train, input_shape, num_classes)

    cnn_model.train_model(X_train, Y_train)

    trained_model = cnn_model.get_model()

    loss, accuracy = trained_model.evaluate(X_test, Y_test)

    print("loss : ", loss)
    print("accuracy : ", accuracy)
