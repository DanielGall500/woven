from flask import Flask, redirect, url_for, render_template, request
from woven.encoder import WOVEncoder
from connections import make_colour_connections, create_graph
from pyvis.network import Network
import numpy as np

app = Flask(__name__)
encoder = WOVEncoder()

def remove_punc(itokens, otokens, encoding):
    test_itokens = np.array(itokens)
    test_encoding = np.array(encoding)
    itokens_punc = np.where(test_itokens == '.')
    for indx in itokens_punc[0]:
        test_itokens = np.delete(test_itokens,indx)
        test_encoding = np.delete(test_encoding,indx,0)

    itokens_punc = np.where(test_itokens == ',')
    for indx in itokens_punc[0]:
        test_itokens = np.delete(test_itokens,indx)
        test_encoding = np.delete(test_encoding,indx,0)

    test_otokens = np.array(otokens)
    otokens_punc = np.where(test_otokens=='.')
    for indx in otokens_punc[0]:
        test_otokens = np.delete(test_otokens,indx)
        test_encoding = np.delete(test_encoding,indx,1)

    otokens_punc = np.where(test_otokens==',')
    for indx in otokens_punc[0]:
        test_otokens = np.delete(test_otokens,indx)
        test_encoding = np.delete(test_encoding,indx,1)

    return test_itokens.tolist(), test_otokens.tolist(), test_encoding

def create_html_table_format(itokens,otokens,encoding):
    table = []
    counter = 0
    first_row = [''] + otokens
    table.append(first_row)
    for i, row in enumerate(encoding):
        new_row = []
        new_row.append(itokens[i])
        for element in row:
            new_row.append(element)
        table.append(new_row)
    return table

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "GET":
        #Load the basic page with no input
        return render_template("index.html", translation="None",\
                is_active=False) 
    else:
        #Our input
        text_input = request.form["inputText"]
        encodable = encoder.encode(text_input)
        encoding = encoder.encoding

        #Main parameters for HTML
        itokens = encodable.merged_inp
        otokens = encodable.t_out
        itokens, otokens, encoding = remove_punc(itokens, otokens, encoding)
        icolours, ocolours = make_colour_connections(encoding)
        out_text = " ".join([x for x in encodable.t_out])

        table_encoding = create_html_table_format(itokens,otokens,encoding)

        #Create the vis
        grph = create_graph(itokens,otokens,icolours,ocolours,encoding)
        grph.write_html('templates/vis.html')

        #Pass to our HTML template
        return render_template("index.html", translation=out_text, itokens=itokens, \
                otokens=otokens, icolours=icolours,\
                ocolours=ocolours, num_input_tokens=range(len(itokens)),\
                num_output_tokens=range(len(otokens)),\
                input_language="English",
                output_language="German",
                is_active=True,
                encoding=table_encoding)

@app.route("/vis", methods=["POST", "GET"])
def vis():
    return render_template("vis.html")

if __name__ == "__main__":
    app.run(debug=True)
