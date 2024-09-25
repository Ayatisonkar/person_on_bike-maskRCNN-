import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage.transform import resize

# Root directory of the project
ROOT_DIR = os.path.abspath("C://mask_RCNN")

print(ROOT_DIR) ################

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class CharacterConfig(Config):
    # Give the configuration a recognizable name
    NAME = "character"

    IMAGES_PER_GPU = 1
    IMAGE_CHANNEL_COUNT=3
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + rick + morty
    if IMAGE_CHANNEL_COUNT == 1:
        MEAN_PIXEL = np.array([123.7])  # Example grayscale mean value
    elif IMAGE_CHANNEL_COUNT == 4:
        MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 0.0])  # Example for RGB-D (4 channels)
    else:
        MEAN_PIXEL = np.array([123.7, 116.8, 103.9]) 
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    ############################################################
    #  Dataset
    ############################################################


class CharacterDataset(utils.Dataset):

    def load_character(self, dataset_dir, subset):

        # Add classes
        self.add_class("character", 1, "person_bike")
        

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        annotations = [a for a in annotations if a['regions']]
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            # Below line is different from hail.py

            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['names'] for s in a ['regions']]
	    
            name_dict = {"person_bike": 1}
            class_ids =[name_dict[a] for a in objects]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            ##changed##
            for i, p in enumerate(polygons):
                all_p_x = np.array(p['all_points_x'])
                all_p_y = np.array(p['all_points_y'])
                all_p_x[all_p_x >= width] = width - 1
                all_p_y[all_p_y >= height] = height - 1
                polygons[i]['all_points_x'] = list(all_p_x)
                polygons[i]['all_points_y'] = list(all_p_y)

            self.add_image(
                "character",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)

    def load_mask(self, image_id,use_mini_mask):

        # If not a objects dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "character":
            return super(self.__class__, self).load_mask(image_id)
        class_ids=image_info['class_ids']

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        resized_mask = np.zeros((56, 56, mask.shape[2]), dtype=np.float32)
        if mask.shape[2] == 0:  # Check if there are no instances
            return np.zeros((self.height, self.width, 0), dtype=np.uint8), np.array([], dtype=np.int32)
        if use_mini_mask:  # Ensure this condition is properly defined
            for i in range(mask.shape[2]):  # Loop through each instance of the mask
                resized_mask[:, :, i] = resize(mask[:, :, i], (56, 56), preserve_range=True, anti_aliasing=True) #Resize to (56, 56, instances)
        if np.count_nonzero(mask) == 0:
            print(f"Warning: Mask for image_id {image_id} is empty.")
        print("info['class_ids'] = ", info['class_ids'])
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "character":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CharacterDataset()
    dataset_train.load_character(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CharacterDataset()
    dataset_val.load_character(args.dataset, "val")
    dataset_val.prepare()

    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='heads')


def color_splash(image, mask):

    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

    #  Training


############################################################

if __name__ == '__main__':
    import argparse

    # Parse command l  line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect character.')
    parser.add_argument("command",
                        metavar='train',
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar= "C://mask_RCNN//dataset",  # in hail it is balloon
                        help='Directory of the character dataset')
    parser.add_argument('--weights', required=True,
                        metavar= r"C:\mask_RCNN\mask_rcnn_coco.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar= r"C:\mask_RCNN\logs",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar= r"C:\mask_RCNN\dataset\val\-1-_jpeg_jpg.rf.182328f592b94bf68f7b1a68ac12551a.jpg",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CharacterConfig()
    else:
        class InferenceConfig(CharacterConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True, exclude=["conv1","mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_mask"])



    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))


		