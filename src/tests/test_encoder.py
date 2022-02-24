import unittest
from woven.encoder import WOVEncoder

class TestWOVEncoder(unittest.TestCase):
    example_str = "I ate it yesterday"

    def test_variation(self):
        encoder = WOVEncoder()
        encodable = encoder.encode(self.example_str)

        variation = encoder.get_variation()
        print(variation)
