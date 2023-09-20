import pytest
import os
import numpy as np
from facerec.dataset_generator import DataGenerator, train_val_test_split


# Test case for DataGenerator data generation
def test_data_generator():
    input_shape = (100, 100, 3)
    batch_size = 32
    seed = 42
    shuffle = True

    # Create a temporary directory for testing
    temp_dir = "test_data/temp_dir"
    os.makedirs(temp_dir, exist_ok=True)

    # Create test CSV files
    with open(os.path.join(temp_dir, "positive_pairs.csv"), "w") as f:
        f.write("name,pair")
        f.write("name1,['image1.jpg', 'image2.jpg']")

    with open(os.path.join(temp_dir, "negative_pairs.csv"), "w") as f:
        f.write("name,name,pair")
        f.write("name1,name2,['image3.jpg', 'image4.jpg']")

    # Create test image files
    os.makedirs(os.path.join(temp_dir, "name1"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "name2"), exist_ok=True)
    with open(os.path.join(temp_dir, "name1", "image1.jpg"), "wb") as f:
        f.write(np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8).tobytes())
    with open(os.path.join(temp_dir, "name1", "image2.jpg"), "wb") as f:
        f.write(np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8).tobytes())
    with open(os.path.join(temp_dir, "name2", "image3.jpg"), "wb") as f:
        f.write(np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8).tobytes())
    with open(os.path.join(temp_dir, "name2", "image4.jpg"), "wb") as f:
        f.write(np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8).tobytes())

    # Initialize DataGenerator
    data_generator = DataGenerator(
        os.path.join(temp_dir, "positive_pairs.csv"),
        os.path.join(temp_dir, "negative_pairs.csv"),
        temp_dir,
        input_shape,
        batch_size,
        seed,
        shuffle
    )

    # Test the __getitem__ method
    batch = data_generator.__getitem__(0)
    assert len(batch) == 2
    assert isinstance(batch[0], tuple)
    assert isinstance(batch[0][0], np.ndarray)
    assert isinstance(batch[0][1], np.ndarray)
    assert isinstance(batch[1], np.ndarray)

    # Clean up
    # os.remove(os.path.join(temp_dir, "positive_pairs.csv"))
    # os.remove(os.path.join(temp_dir, "negative_pairs.csv"))
    # os.remove(os.path.join(temp_dir, "name1", "image1.jpg"))
    # os.remove(os.path.join(temp_dir, "name1", "image2.jpg"))
    # os.remove(os.path.join(temp_dir, "name2", "image3.jpg"))
    # os.remove(os.path.join(temp_dir, "name2", "image4.jpg"))
    # os.rmdir(os.path.join(temp_dir, "name1"))
    # os.rmdir(os.path.join(temp_dir, "name2"))
    # os.rmdir(temp_dir)
    # Clean up temporary images after the tests
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(temp_dir)


# Test case for train_val_test_split function
def test_train_val_test_split():
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1
    seed = 42

    # Create a temporary directory for testing
    temp_dir = "test_data/temp_dir2"
    os.makedirs(temp_dir, exist_ok=True)

    # Create a test pairs file
    with open(os.path.join(temp_dir, "all_pairs.txt"), "w") as f:
        f.write("header\n")
        for i in range(100):
            f.write(f"data{i}\n")

    # Call the function
    train_val_test_split(os.path.join(temp_dir, "all_pairs.txt"), train_size, val_size, test_size, seed)

    # Check if the split files exist
    assert os.path.isfile(os.path.join(temp_dir, "train_all_pairs.txt"))
    assert os.path.isfile(os.path.join(temp_dir, "val_all_pairs.txt"))
    assert os.path.isfile(os.path.join(temp_dir, "test_all_pairs.txt"))

    # Clean up
    # os.remove(os.path.join(temp_dir, "all_pairs.txt"))
    # os.remove(os.path.join(temp_dir, "train_all_pairs.txt"))
    # os.remove(os.path.join(temp_dir, "val_all_pairs.txt"))
    # os.remove(os.path.join(temp_dir, "test_all_pairs.txt"))
    # os.rmdir(temp_dir)
    # Clean up temporary images after the tests
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(temp_dir)


# Run the tests
if __name__ == '__main__':
    pytest.main([__file__])
