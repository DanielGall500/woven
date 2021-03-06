import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
import re
"""
WOVEncodable
A matrix of normalised Shapley values for a particular
translation from X to Y.
"""
class WOVEncodable:
    def __init__(self, org_input, t_inp, t_out, shap_values):
        #Check whether encodable shape is
        #correct
        if self._is_correct_shape(t_inp, t_out, shap_values):
            self.org_inp = org_input
            self.t_inp = t_inp
            self.t_out = t_out
            self.raw = shap_values
            self.merged_inp = self._split_by_word(org_input)
        else:
            raise Exception("Arguments Of Invalid Shape")

    """
    A confidence score measures how certain we are that a
    particular input feature i has relevance to a particular
    output feature o, where the maximum value of 100 is allocated
    to the i-o connection we are most confident in.

    Confidence scores take into account all positive feature
    contributions. For each contribution, their contribution
    to a given output o is calculated when compared to their
    contributions to all other outputs O. A higher score means
    greater relative importance to a particular output.

    These are then squeezed in the range [0,1] and multiplied
    by 100 in order to make them more easily interpretable.
    
    Each row corresponds to an output G, and each column an input E:
       
        __E1_E2_E3__
    G1 | ...
    G2 |
    G3 |
    ...
    """
    def get_confidence_scores(self):
        I_size = len(self.raw)
        O_size = len(self.raw[0])

        confidence_scores = []

        for o_indx in range(O_size):
            #Column vector of inputs I on single output o
            shap_Io = self.get_output_col(o_indx)
            shap_Io_adjusted = []

            #Iterate through each input i on o
            for i_indx, i in enumerate(shap_Io):
                if i > 0:
                    #Get all shap outputs O for this input i
                    shap_iO = self.get_input_row(i_indx)
                    
                    #Find the average contribution of
                    #i to all other outputs in O except o
                    #We only take into account positive
                    #contributions, as these are the ones
                    #fighting to say 'I caused this!'
                    shap_iO[o_indx] = -1
                    shap_iO = [shap for shap in shap_iO if shap>0]
                    if len(shap_iO) > 0:
                        avg = np.sum(shap_iO)/(len(shap_iO))
                    else:
                        avg = 0

                    #Does i hold it's own for this o compared
                    #to other o's?
                    i_adjusted = i - avg
                else:
                    #Ensure negative Shapley values
                    #have min confidence by assigning
                    #nan and giving min after
                    i_adjusted = np.nan
                shap_Io_adjusted.append(i_adjusted)

            confidence_scores.append(shap_Io_adjusted)

        #Set irrelevant Shapley values from above
        #equal to minimum adjusted value
        #This gives them a confidence of 0
        confidence_scores = np.array(confidence_scores)
        min_score = np.nanmin(confidence_scores)
        confidence_scores = np.nan_to_num(confidence_scores, nan=min_score-1) 

        #MinMax Normalisation
        #Add the min onto all scores to bring them 0 or above
        #NaNs (ie irrelevant Shapley vals) will be equal to -1.
        confidence_scores = np.add(confidence_scores, abs(min_score))
        
        #Find the new max
        max_score = np.amax(confidence_scores)

        #Divide everything by this max
        confidence_scores = np.divide(confidence_scores, max_score)

        #Multiply for simplicity to create range (0,100)
        confidence_scores = np.multiply(confidence_scores,100)

        #Set all irrelevant Shapley values to -1
        confidence_scores[confidence_scores < 0] = -1

        #Round the scores
        confidence_scores = np.around(confidence_scores, 2)

        #Transpose back to original form
        confidence_scores = np.transpose(confidence_scores)
        return confidence_scores
        
    def get_encoding(self, l=20, theta=10, merged=False):
        confidence_scores = self.get_confidence_scores()
        #Transpose for simplicity. Reverted at the end
        confidence_scores = np.transpose(confidence_scores)
        cs_shape = confidence_scores.shape
        wov_encoding = np.zeros(cs_shape)

        #Iterate row of English contributions
        #to a single German output
        for row_indx, row in enumerate(confidence_scores):
            max_val = np.amax(row)

            #If we can make at least one connection
            #in this row then continue
            if max_val >= l:
                #Iterate I->O contributions
                for col_indx, val in enumerate(row):
                    #Check if value is irrelevant
                    if val >= 0:
                        #Calculate the lower bound for this row
                        lower_bound = max_val-theta if (max_val-theta >= 0) else 0

                        #Does this value meet the lower bound?
                        wov_encoding[row_indx][col_indx] = 1 \
                                if val >= lower_bound else 0
        wov_encoding = np.transpose(wov_encoding)
    
        if merged:
            return self._detokenise(wov_encoding)
        else:
            return wov_encoding

    def _detokenise(self, enc, tokenised_input=None, \
            tokenised_output=None, original_input=None):
        tokenised_input = self.t_inp if tokenised_input is None \
                else tokenised_input
        tokenised_output = self.t_out if tokenised_output is None \
                else tokenised_output
        original_input = self.org_inp if original_input is None \
                else original_input

        vector_sum = np.zeros([len(tokenised_output)])

        words = self.merged_inp
        merged_matrix = []
        next_word = True
        curr = -1
        for t, t_encoding in zip(tokenised_input, enc):
            if next_word:
                #Store merged vector in matrix
                vector_sum = t_encoding

                #Move onto the next word
                curr += 1

                #If we have reached end of our
                #merged words, ignore the rest
                #of the original tokens
                if curr >= len(words):
                    break
                word = words[curr]
            else:
                #Merge last vector and this vector
                vector_sum += t_encoding

            #Remove token from word and see what's next
            #Remove _ from tokenised words
            if len(t) > 0:
                if t not in word:
                    t = t[1:]

            word = word.replace(t, '')
            next_word = (word == '')
            if next_word:
                merged_matrix.append(vector_sum)

        merged_matrix = np.array(merged_matrix)
        merged_matrix = np.where(merged_matrix > 0, 1, 0)
        return merged_matrix 
        
    def _split_by_word(self, I: str):
        punc = ['\s', '\,\s','\.\s','\?\s']
        delims = '|'.join([p for p in punc])
        return re.split(delims, I)

    def index_in_input_of(self, A: str):
        return self.t_inp.index(A)

    def index_in_output_of(self, A: str):
        return self.t_out.index(A)

    def shap_contribution_of_A_to_B(self, A: str, B: str):
        if not self.exists_in_input(A) or \
        not self.exists_in_output(B):
            raise Exception("Arguments Invalid")

        A_index = self.index_in_input_of(A)
        B_index = self.index_in_output_of(B)

        return self.get_shap_value(A_index, B_index)

    def get_shap_value(self, i, j):
        return self.raw[i][j].copy()

    def get_input_row(self, i):
        return self.raw[i].copy()

    def get_output_col(self, j):
        return [row[j] for row in self.raw].copy()

    def exists_in_input(self, A: str):
        return (A in self.t_inp)

    def exists_in_output(self, A: str):
        return (A in self.t_out)

    def _is_correct_shape(self, t_inp, t_out, shap_values):
        m = len(t_inp)
        n = len(t_out)
        mn = np.prod(shap_values.shape)
        return (m*n == mn)
