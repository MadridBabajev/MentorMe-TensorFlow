import os
import cv2
import tensorflow as tf
import random

class IAMDataset(tf.keras.utils.Sequence):
    """
    A simple Sequence (or Generator) that loads images + labels from the IAM-style dataset.
    """
    def __init__(self, lines_file, base_dir, batch_size=32, img_height=128, img_width=None, max_length=64):
        self.lines = []
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.max_length = max_length
    
        with open(lines_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
    
                fields = line.split()
                image_id = fields[0]    # e.g. 'a01-000u-s00-00'
                # Reconstruct the transcription
                # Usually fields[8:] holds the tokens; adjust if your file is different
                transcription_str = " ".join(fields[8:]).replace('|', ' ')
    
                # Build the path to the .png
                # example: 'a01', '000u', 's00', '00' => subfolder='a01', folder_name='a01-000u'
                parts = image_id.split('-')  # e.g. ['a01','000u','s00','00']
                if len(parts) < 3:
                    continue
    
                subfolder = parts[0]              # e.g. 'a01'
                folder_name = "-".join(parts[:2]) # e.g. 'a01-000u'
                image_name = image_id + ".png"    # 'a01-000u-s00-00.png'
                full_path = os.path.join(
                    base_dir,
                    subfolder,
                    folder_name,
                    image_name
                )
    
                # If the file actually exists, store it
                if os.path.isfile(full_path):
                    self.lines.append((full_path, transcription_str))
        # Sort the resulting list of (img_path, transcription)
        # random.shuffle(self.lines) # Randomly
        self.lines.sort(key=lambda x: x[0]) # Alphabetically

    def __len__(self):
        return len(self.lines) // self.batch_size

    def __getitem__(self, idx):
        batch_lines = self.lines[idx*self.batch_size : (idx+1)*self.batch_size]

        images = []
        labels = []
        paths = []

        for (img_path, label_str) in batch_lines:
            # Read grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Scale img preserving the aspect ratio
            original_h, original_w = img.shape
            desired_h = self.img_height
            scale = desired_h / original_h
            new_w = int(original_w * scale)
            resized = cv2.resize(img, (new_w, desired_h))
            
            # Convert to float32 and add channel dim
            resized_tensor = tf.constant(resized, dtype=tf.float32)[..., tf.newaxis]

            # Pad to max width to get a fixed-size tensor
            if self.img_width is not None:
                # if new width < desired width => pad on the right
                if new_w < self.img_width:
                    padded = tf.pad(
                        resized_tensor,
                        paddings=[[0,0], [0, self.img_width - new_w], [0,0]],
                        mode='CONSTANT',
                        constant_values=255.0  # white background
                    )
                    final = padded
                # if new width > desired width => crop on the right
                elif new_w > self.img_width:
                    final = resized_tensor[:, :self.img_width, :]
                else:
                    # exact match
                    final = resized_tensor
            else:
                # If self.img_width is None, just use resized_tensor as is
                final = resized_tensor
                
            final = final / 255.0            
            images.append(final)
            labels.append(label_str)
            paths.append(img_path)

        # Convert to TF tensors
        x_batch = tf.stack(images, axis=0)

        return x_batch, labels, paths
