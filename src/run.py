from flask import Flask, redirect, url_for, render_template, request
from woven.encoder import WOVEncoder

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
        out_text = " ".join([x for x in encodable.t_out])
        return render_template("index.html", translation=out_text)
if __name__ == "__main__":
    app.run(debug=True)
