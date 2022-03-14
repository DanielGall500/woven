from flask import Flask, redirect, url_for, render_template, request
from woven.encoder import WOVEncoder
from connections import make_colour_connections

app = Flask(__name__)
encoder = WOVEncoder()

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "GET":
        return render_template("index.html", translation="None") 
    else:
        print(request.form.keys())
        text_input = request.form["inputText"]
        encodable = encoder.encode(text_input)
        encoding = encoder.encoding
        icolours, ocolours = make_colour_connections(encoding)
        out_text = " ".join([x for x in encodable.t_out])
        itokens = encodable.merged_inp
        otokens = encodable.t_out
        return render_template("index.html", translation=out_text, itokens=itokens, \
                otokens=otokens, icolours=icolours,\
                ocolours=ocolours, num_input_tokens=range(len(itokens)),\
                num_output_tokens=range(len(otokens)))

if __name__ == "__main__":
    app.run(debug=True)
