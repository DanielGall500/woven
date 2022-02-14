import unittest
import numpy as np
from woven.encodable import WOVEncodable

class TestEncodable(unittest.TestCase):

    ex_input = ["This", "is", "a", "test"]
    ex_output = ["Das", "ist", "ein", "Test"]
    ex_shap = np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]])

    #Test that values stored correctly
    def test_is_correct_shape(self):
        enc = WOVEncodable(self.ex_input, self.ex_output, self.ex_shap)

        self.assertTrue(np.array_equal(enc.t_inp, self.ex_input))
        self.assertTrue(np.array_equal(enc.t_out, self.ex_output))
        self.assertTrue(np.array_equal(enc.raw, self.ex_shap))

    def test_confidence_scores(self):
        score_test_shap_vals = np.array([
            [5, 1, 1, 1],
            [1, 5, -1, 1],
            [1, 1,  5, 1],
            [1, 1,  1, 10]])

        enc = WOVEncodable(self.ex_input, self.ex_output, \
                score_test_shap_vals)
        cs = enc.get_confidence_scores()

        correct_cs = np.array([
            [58.33, 8.33, 13.89, 0],
            [13.89, 58.33, 13.89, 0],
            [13.89, -1, 58.33, 0],
            [13.89, 8.33, 13.89, 100]])
        self.assertTrue(np.array_equal(cs, correct_cs))
