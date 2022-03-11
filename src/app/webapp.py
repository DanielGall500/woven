from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "GET":
        return render_template("index.html", translation="None") 
    else:
        print(request.form.keys())
        text_input = request.form["inputText"]
        return render_template("index.html", translation=text_input)
if __name__ == "__main__":
    app.run(debug=True)
