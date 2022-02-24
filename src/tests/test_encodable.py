import unittest
import numpy as np
from woven.encodable import WOVEncodable

class TestEncodable(unittest.TestCase):

    org_input = "This is a test"
    ex_input = ["This", "is", "a", "test"]
    ex_output = ["Das", "ist", "ein", "Test"]

    ex_shap_vals = np.array([
        [5, 1, 1, 1],
        [1, 5, -1, 1],
        [1, 1,  5, 1],
        [1, 1,  1, 10]])

    ex_conf_scores = np.transpose(np.array([
        [58.33, 8.33, 13.89, 0],
        [13.89, 58.33, 13.89, 0],
        [13.89, -1, 58.33, 0],
        [13.89, 8.33, 13.89, 100]]))

    ex_encoding = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]])
    
    #Test that values stored correctly
    def test_is_correct_shape(self):
        enc = WOVEncodable("",self.ex_input, self.ex_output, self.ex_shap_vals)

        self.assertTrue(np.array_equal(enc.t_inp, self.ex_input))
        self.assertTrue(np.array_equal(enc.t_out, self.ex_output))
        self.assertTrue(np.array_equal(enc.raw, self.ex_shap_vals))

    def test_confidence_scores(self):
        enc = WOVEncodable("",self.ex_input, self.ex_output, \
                self.ex_shap_vals)
        cs = enc.get_confidence_scores()

        self.assertTrue(np.array_equal(cs, self.ex_conf_scores))

    def test_wov_encoding(self):
        enc = WOVEncodable("",self.ex_input, self.ex_output, \
                self.ex_shap_vals)

        wov_encoding_one = enc.get_encoding(l=20, theta=10)
        wov_encoding_two = enc.get_encoding(\
                l=10, theta=49)

        #Encoding for lambda=20, theta=10
        correct_encoding_one = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        correct_encoding_two = np.transpose(correct_encoding_one)
        
        #Encoding for lambda=10, theta=49
        correct_encoding_two = np.array([
            [1, 0, 1, 0],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1]])
        correct_encoding_two = np.transpose(correct_encoding_two)

        self.assertTrue(np.array_equal(wov_encoding_one, correct_encoding_one))
        self.assertTrue(np.array_equal(wov_encoding_two, correct_encoding_two))

    def test_merging(self):
        enc = WOVEncodable(self.org_input, self.ex_input, self.ex_output, \
                self.ex_shap_vals)
        merged_matrix_1 = enc.get_encoding(l=20, theta=10, merged=True)
        self.assertTrue(np.array_equal(merged_matrix_1, self.ex_encoding))

        ex2_org_str = "I ate it"
        ex2_tinp = ["I", "at", "e", "it"]
        ex2_tout = ["Ich", "habe", "es", "gegessen"]
        ex2_enc = np.array([ \
                [1,1,0,0],
                [0,1,0,1],
                [0,1,0,1],
                [0,0,1,0]])

        ex2_correct_merge = np.array([
            [1,1,0,0],
            [0,1,0,1],
            [0,0,1,0]])
        enc.merged_inp = np.array(["I", "ate", "it"])
        merged_matrix = enc._detokenise(ex2_enc, ex2_tinp, ex2_tout, ex2_org_str) 
        self.assertTrue(np.array_equal(merged_matrix, ex2_correct_merge))
