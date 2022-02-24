import shap
from woven.encodable import WOVEncodable
import numpy as np

"""
WOVEncoder
Input: A non-tokenised English sentence E.
Purpose: Runs E through the SHAP Partition Explainer and wraps
the resulting Shapley values in a WOVEncodable object.
"""

class WOVEncoder:
    explainer_file_path = "woven/explainer/partition_explainer.pickle"
    encodable = None
    
    def __init__(self):
        #Load the Pre-trained SHAP Partition Explainer
        with open(self.explainer_file_path, 'rb+') as exp_file:
            self.explainer = shap.Explainer.load(exp_file)


    def encode(self, E: str, \
            min_confidence_threshold=20, \
            multi_confidence_threshold=10, \
            merge_tokens=True):
        if self._is_valid_input(E):
            shap_exp = self.explainer([E])
        else:
            raise Exception("Invalid Input")

        #Save relevant information to encode word order
        input_tokenised = shap_exp.feature_names[0]
        output_tokenised = shap_exp.output_names
        
        shap_values = shap_exp.values
        shap_values = np.squeeze(shap_values)
        
        #Store relevant information in encodable object
        self.encodable = WOVEncodable(org_input=E,
                t_inp=input_tokenised,
                t_out=output_tokenised,
                shap_values=shap_values)
        
        self.encoding = self.encodable.get_encoding(\
                l=min_confidence_threshold,\
                theta=multi_confidence_threshold,\
                merged=merge_tokens)
        
        return self.encodable

    def get_variation(self):
        if self.encodable is None:
            return None

        enc = self.encoding
        return enc

    def _is_valid_input(self, E):
        return True

if __name__ == "__main__":
    main()
