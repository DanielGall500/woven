import shap
from encodable import WOVEncodable
import numpy as np

"""
WOVEncoder
Input: A non-tokenised English sentence E.
Purpose: Runs E through the SHAP Partition Explainer and wraps
the resulting Shapley values in a WOVEncodable object.
"""

class WOVEncoder:
    explainer_file_path = "explainer/partition_explainer.pickle"

    def __init__(self):
        #Load the Pre-trained SHAP Partition Explainer
        with open(self.explainer_file_path, 'rb+') as exp_file:
            self.explainer = shap.Explainer.load(exp_file)


    def encode(self, E: str):
        if self._is_valid_input(E):
            shap_exp = self.explainer(E)

        #Save relevant information to encode word order
        input_tokenised = shap_exp.feature_names[0]
        output_tokenised = shap_exp.output_names
        
        shap_values = shap_exp.values
        shap_values = np.squeeze(shap_values)
        
        #Store relevant information in encodable object
        encodable = WOVEncodable(t_inp=input_tokenised,
                t_out=output_tokenised,
                shap_values=shap_values)

        return encodable


    def _is_valid_input(self, E):
        return True

def main():
    encoder = WOVEncoder()
    example_str = "I hope that it is good thanks"
    encoded = encoder.encode([example_str])
    first_word_input = encoded.t_inp[1]
    first_word_output = encoded.t_out[2]
    print("SHAP Values")
    print(np.around(encoded.raw,2))
    cv = encoded.get_confidence_scores()
    print("--Confidence Valyes--")
    print(cv)
    print("Rows (eng): ", encoded.t_inp)
    print("Cols (deu): ", encoded.t_out)
    wov = encoded.get_wov_encoding()
    print("--WOV Encoding--")
    print(wov)

if __name__ == "__main__":
    main()
