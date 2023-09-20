# import pytest
# import os
# import numpy as np
# from facerec.pairs_generator import PairsGenerator


# # Test case for PairsGenerator
# def test_pairs_generator():
#     # Create a temporary directory for testing
#     temp_dir = "test_data/temp_dir"
#     os.makedirs(temp_dir, exist_ok=True)

#     # # Create test CSV files
#     # with open(os.path.join(temp_dir, "positive_pairs.csv"), "w") as f:
#     #     f.write("name,pair")
#     #     f.write("name1,['image1.jpg', 'image2.jpg']")

#     # with open(os.path.join(temp_dir, "negative_pairs.csv"), "w") as f:
#     #     f.write("name,name,pair")
#     #     f.write("name1,name2,['image3.jpg', 'image4.jpg']")

#     # Create test image files
#     os.makedirs(os.path.join(temp_dir, "name1"), exist_ok=True)
#     os.makedirs(os.path.join(temp_dir, "name2"), exist_ok=True)
#     with open(os.path.join(temp_dir, "name1", "image1.jpg"), "wb") as f:
#         f.write(np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8).tobytes())
#     with open(os.path.join(temp_dir, "name1", "image2.jpg"), "wb") as f:
#         f.write(np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8).tobytes())
#     with open(os.path.join(temp_dir, "name2", "image3.jpg"), "wb") as f:
#         f.write(np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8).tobytes())
#     with open(os.path.join(temp_dir, "name2", "image4.jpg"), "wb") as f:
#         f.write(np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8).tobytes())

#     gen = PairsGenerator(images_directory=temp_dir)

#     gen.generate_positive_pairs()
#     assert len(gen.positive_pairs) == 1 # Assuming only 2 images per person
#     gen.export_positive_pairs_to_csv(filename='test_positive_pairs.csv')
#     # Check if the CSV file was created
#     assert os.path.exists(os.path.join(gen.pairs_directory, 'test_positive_pairs.csv'))

#     gen.generate_negative_combinations(combinations_num=2)  # Generate 2 combinations
#     assert len(gen.negative_pairs) == 2
#     gen.export_negative_pairs_to_csv(filename='test_negative_pairs.csv')
#     # Check if the CSV file was created
#     assert os.path.exists(os.path.join(gen.pairs_directory, 'test_negative_pairs.csv'))


#     # Clean up temporary images after the tests
#     for root, dirs, files in os.walk(temp_dir, topdown=False):
#         for file in files:
#             os.remove(os.path.join(root, file))
#         for dir in dirs:
#             os.rmdir(os.path.join(root, dir))
#     os.rmdir(temp_dir)
