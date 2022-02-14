import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
"""
WOVEncodable
A matrix of normalised Shapley values for a particular
translation from X to Y.
"""
class WOVEncodable:
    def __init__(self, t_inp, t_out, shap_values):
        #Check whether encodable shape is
        #correct
        if self._is_correct_shape(t_inp, t_out, shap_values):
            self.t_inp = t_inp
            self.t_out = t_out
            self.raw = shap_values

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
        return confidence_scores
                
    def get_wov_encoding(self, binary=False, n=2):
        cs = self.get_confidence_scores()
        wov_encoding = np.zeros(cs.shape)

        #Iterate through each row of values E 
        #on a given g
        for i, row in enumerate(cs):
            sorted_row = np.sort(row)

            #Find the number of highest scores n
            for j in range(n):
                nth_best_val = sorted_row[len(row)-1-j]
                nth_best_indx = np.where(row == nth_best_val)[0][0]

                #If we have selected binary, give a binary encoding.
                #Otherwise, include the confidence score.
                wov_encoding[i][nth_best_indx] = 1 if binary else nth_best_val
        return wov_encoding

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
