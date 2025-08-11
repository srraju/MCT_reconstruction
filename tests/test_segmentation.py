import unittest
from mct_segmentation.segmentation import ImageHandler

class TestSegmentation(unittest.TestCase):
    def test_initialization(self):
        handler = ImageHandler('some_folder', ['a.tif', 'b.tif'], 20)
        self.assertEqual(handler.z_height, 20)

if __name__ == "__main__":
    unittest.main()
