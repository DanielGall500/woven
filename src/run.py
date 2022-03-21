from flask import Flask, redirect, url_for, render_template, request
from woven.encoder import WOVEncoder
from connections import make_colour_connections, create_graph
from pyvis.network import Network

app = Flask(__name__)
encoder = WOVEncoder()

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
        icolours, ocolours = make_colour_connections(encoding)
        out_text = " ".join([x for x in encodable.t_out])
        itokens = encodable.merged_inp
        otokens = encodable.t_out
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
