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

    def get_confidence_scores(self):
        I_size = len(self.raw)
        O_size = len(self.raw[0])
        print(I_size)
        print(O_size)

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
                        print("For {}".format(i))
                        print(shap_iO)
                        avg = np.sum(shap_iO)/(len(shap_iO))
                    else:
                        avg = 0

                    #Does i hold it's own for this o compared
                    #to other o's?
                    i_adjusted = i - avg
                else:
                    i_adjusted = 0
                shap_Io_adjusted.append(i_adjusted)

            confidence_scores.append(shap_Io_adjusted)

        #MinMax Normalisation
        #for indx, cs_o in enumerate(confidence_scores):
        #    normed = normalize(cs_o, norm='max')
        #            confidence_scores[indx] = normed
        
        print("Unnormed:")
        for row in confidence_scores:
            print(np.around(row,2))
        print("----")
        #confidence_scores = normalize(confidence_scores, norm='max', axis=0)
        #MinMax Norm
        #Find the min
        confidence_scores = np.array(confidence_scores)
        min_score = np.amin(confidence_scores)

        #Add the min onto it
        confidence_scores = np.add(confidence_scores, abs(min_score))

        #Find the new max
        max_score = np.amax(confidence_scores)

        #Divide everything by this max
        confidence_scores = np.divide(confidence_scores, max_score)

        #Multiply for simplicity to create range (0,100)
        confidence_scores = np.multiply(confidence_scores,100)

        #Round the scores
        confidence_scores = np.around(confidence_scores, 2)
        return confidence_scores
                

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
        return self.raw[i][j]

    def get_input_row(self, i):
        return self.raw[i]

    def get_output_col(self, j):
        return [row[j] for row in self.raw]

    def exists_in_input(self, A: str):
        return (A in self.t_inp)

    def exists_in_output(self, A: str):
        return (A in self.t_out)

    def _is_correct_shape(self, t_inp, t_out, shap_values):
        m = len(t_inp)
        n = len(t_out)
        mn = np.prod(shap_values.shape)
        return (m*n == mn)
