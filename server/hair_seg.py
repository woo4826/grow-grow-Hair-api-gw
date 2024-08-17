import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



class HairSegmenter:
    def __init__(self, desired_height=480, desired_width=480):
        self.desired_height = desired_height
        self.desired_width = desired_width

    def resize_and_show(self, image, window_name="Image"):
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (self.desired_width, math.floor(h / (w / self.desired_width))))
        else:
            img = cv2.resize(image, (math.floor(w / (h / self.desired_height)), self.desired_height))
        # cv2.imshow(window_name, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    def most_frequent_colour(self, image_file_name):
        image = cv2.imread(image_file_name)

        # Reshape image to 2D array of pixels
        pixels = image.reshape(-1, 3)

        # Apply the skin color conditions
        skin_pixels = []
        for pixel in pixels:
            B, G, R = pixel  # Note: OpenCV reads images in BGR format
            if (
                R > 95 and G > 40 and B > 20 and
                (max(R, G, B) - min(R, G, B)) > 15 and
                abs(R - G) > 15 and R > G and R > B
            ):
                skin_pixels.append(pixel)

        if skin_pixels:
            skin_pixels = np.array(skin_pixels, dtype=np.float32)
            n_colors = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(skin_pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            dominant = palette[np.argmax(counts)]  # Most frequent skin tone in BGR
        else:
            # Fallback: If no skin tones are found, return a default color or raise an error
            dominant = np.array([0, 0, 0], dtype=np.float32)  # or raise an error

        return dominant

    def segment_hair(self, image_file_name):
        self.flesh_tone_color = self.most_frequent_colour(image_file_name)

        base_options = python.BaseOptions(model_asset_path="hair_models/hair_segmentation.tflite")
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            ori_images = cv2.imread(image_file_name, cv2.IMREAD_UNCHANGED)
            mp_image = mp.Image.create_from_file(image_file_name)
            #
            # Apply Gaussian Blur to reduce noise
            blurred_image = cv2.GaussianBlur(ori_images, (5, 5), 5)

            # Apply Median Filter to further reduce noise
            filtered_image = cv2.medianBlur(blurred_image, 5)
            #
            segmentation_result = segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask

            image_data = mp_image.numpy_view()
            h, w = image_data.shape[:2]
            num_channels = image_data.shape[2]

            if num_channels == 4:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB)
            elif num_channels == 1:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)

            flesh_tone_image = np.zeros((h, w, 3), dtype=np.uint8)
            flesh_tone_image[:, :, :] = self.flesh_tone_color
            if ori_images.shape[2] == 4:  # If ori_images has an alpha channel, remove it
                ori_images = cv2.cvtColor(ori_images, cv2.COLOR_RGBA2RGB)
            
            
            if filtered_image.shape[2] == 4:  # If ori_images has an alpha channel, remove it
                filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_RGBA2RGB)
            
            category_mask_np = category_mask.numpy_view()
            condition = np.stack((category_mask_np,) * 3, axis=-1) > 0.2

            # Display original image and flesh-tone image for comparison
            self.resize_and_show(ori_images, window_name="Original Image")
            # self.resize_and_show(flesh_tone_image, window_name="Flesh-tone Image")

            output_image = np.where(condition, flesh_tone_image, ori_images)

            print(f"Flesh-tone image of {image_file_name}:")
            self.resize_and_show(output_image, window_name=f"Flesh-tone Image of {image_file_name}")
            return output_image




