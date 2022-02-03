import cv2
import os


class ImageWriter:
    def __init__(self, images, image_labels, image_paths, output_dir):
        self.images = images
        self.image_labels = image_labels
        self.image_paths = image_paths
        self.output_dir = output_dir
        self._imwrite()

    def _imwrite(self):
        print(self.images.shape)
        for index, image in enumerate(self.images):
            output_path = os.path.join(self.output_dir, self.image_labels[index], self.image_paths[index])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(output_path)
            cv2.imwrite(output_path, image)
        print('Saved Reconstructed Image')
