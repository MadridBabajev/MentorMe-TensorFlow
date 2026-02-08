import sys
import os

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, project_root)

from src.utils.dataset import IAMDataset

def representative_data_gen(lines_file, base_dir, sample_count=100):
    """
    Create a generator that yields a small number of image samples.
    These samples will be used by TFLite for int8 calibration.
    
    :param lines_file: path to your 'sentences.txt' or similar lines file
    :param base_dir: root folder with image data
    :param sample_count: how many samples to yield
    """
    # Use a small batch_size=1 or so for calibration
    calibration_batch_size = 1

    # Initialize your IAMDataset
    dataset = IAMDataset(
        lines_file=lines_file,
        base_dir=base_dir,
        batch_size=calibration_batch_size,
        img_height=128,
        img_width=800
    )

    # Make sure we don't exceed dataset length
    max_steps = min(sample_count, len(dataset))

    # Now iterate and yield small batches
    for step in range(max_steps):
        # x_batch will have shape (1, 128, 800, 1) if batch_size=1
        x_batch, _, _ = dataset[step]
        # TFLite expects a list (or tuple) of input arrays.
        # Even if the model has only one input, it should be [x_batch].
        yield [x_batch]
