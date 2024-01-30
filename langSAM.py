from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import numpy as np


class LangSAMProcessor:
    """
    Processor class for handling image inference using the LangSAM model.

    Attributes:
        model (LangSAM): An instance of the LangSAM model for image processing.
        text_prompt (str): The text prompt used for the inference.
    """

    def __init__(self, text_prompt):
        """
        Initialize the LangSAM processor.

        Args:
            text_prompt (str): The text prompt for model inference.
        """
        self.model = LangSAM()
        self.text_prompt = text_prompt


    def inference(self, rgb_image_array, single_mask=False, visualize_info=True):
        """
        Perform inference on an RGB image array. Optionally return only the mask with the highest confidence.

        Args:
            rgb_image_array (np.ndarray): The RGB image array for inference.
            single_mask (bool): If True, only the mask with the highest confidence is processed and returned.

        Returns:
            np.ndarray or list: The numpy array representing the mask with the highest confidence if single_mask is True,
                                otherwise a list of numpy arrays representing all masks.
        """
        image_pil = Image.fromarray(rgb_image_array)
        masks, boxes, phrases, logits = self.model.predict(image_pil, self.text_prompt)

        if len(masks) == 0:
            rospy.loginfo(f"No objects of the '{self.text_prompt}' prompt detected in the image.")
            return None

        if single_mask:
            # Process only the mask with the highest confidence
            max_confidence_index = np.argmax([logit.item() for logit in logits])
            mask = masks[max_confidence_index].squeeze().cpu().numpy()
            if visualize_info:
                self.display_image_with_masks(image_pil, [mask])
                self.print_bounding_boxes([boxes[max_confidence_index]])
                self.print_detected_phrases([phrases[max_confidence_index]])
                self.print_logits([logits[max_confidence_index]])
            return mask
        else:
            # Process all masks
            masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
            if visualize_info:
                self.display_image_with_masks(image_pil, masks_np)
                self.print_bounding_boxes(boxes)
                self.print_detected_phrases(phrases)
                self.print_logits(logits)
            return masks_np


    @staticmethod
    def display_image_with_masks(image, masks):
        """Display the original image and masks side by side."""
        num_masks = len(masks)
        fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        for i, mask_np in enumerate(masks):
            axes[i+1].imshow(mask_np, cmap='gray')
            axes[i+1].set_title(f"Mask {i+1}")
            axes[i+1].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_image_with_boxes(image, boxes, logits):
        """Display the original image and boxes side by side."""
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title("Image with Bounding Boxes")
        ax.axis('off')

        for box, logit in zip(boxes, logits):
            x_min, y_min, x_max, y_max = box
            confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
            box_width = x_max - x_min
            box_height = y_max - y_min

            # Draw bounding box
            rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            # Add confidence score as text
            ax.text(x_min, y_min, f"Confidence: {confidence_score}", fontsize=8, color='red', verticalalignment='top')

        plt.show()

    @staticmethod
    def print_bounding_boxes(boxes):
        print("Bounding Boxes:")
        for i, box in enumerate(boxes):
            print(f"Box {i+1}: {box}")

    @staticmethod
    def print_detected_phrases(phrases):
        print("\nDetected Phrases:")
        for i, phrase in enumerate(phrases):
            print(f"Phrase {i+1}: {phrase}")

    @staticmethod
    def print_logits(logits):
        print("\nConfidence:")
        for i, logit in enumerate(logits):
            print(f"Logit {i+1}: {logit}")



if __name__ == '__main__':
    import rospy
    import ros_numpy
    from sensor_msgs.msg import Image as ImageMsg
    from PoseEst.direct.preprocessor import Preprocessor, pose_inv, SceneData

    rospy.init_node('LangSAM', anonymous=True)
    langSAMProcessor = LangSAMProcessor(text_prompt="lego")

    rgb_message = rospy.wait_for_message("camera/color/image_raw", ImageMsg)
    depth_message = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", ImageMsg)
    
    live_rgb = ros_numpy.numpify(rgb_message)
    live_depth = ros_numpy.numpify(depth_message)

    assert live_rgb.shape[0] == live_depth.shape[0] == 720

    live_mask = langSAMProcessor.inference(live_rgb, single_mask=True, visualize_info=False)
    if live_mask is None:
        rospy.loginfo("No valid masks returned from inference.")
        raise ConnectionAbortedError

    demo_rgb = np.array(Image.open("data/lego/demo_rgb.png"))
    demo_depth = np.array(Image.open("data/lego/demo_depth.png"))
    demo_mask = np.array(Image.open("data/lego/demo_mask.png"))
    intrinsics = np.load("handeye/intrinsics.npy")

    data = SceneData(
        image_0=demo_rgb,
        image_1=live_rgb,
        depth_0=demo_depth,
        depth_1=live_depth,
        seg_0=demo_mask,
        seg_1=(live_mask * 255).astype(np.uint8),
        intrinsics_0=intrinsics,
        intrinsics_1=intrinsics,
        T_WC=np.eye(4) # cam frame
    )

    processor = Preprocessor()
    data.update(processor(data))


    

