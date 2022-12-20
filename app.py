from flask import Flask, render_template, request, jsonify
import yaml
from src.models.predict_model import predict_value


def retrieve_params(config):
    with open(config) as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params

params = retrieve_params(config="params.yaml")

app = Flask(__name__,
            static_folder=params["webapp"]["static_folder"],
            template_folder=params["webapp"]["template_folder"])


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("home.html")
    else:
        if request.form:
            response_ = predict_value(request.form)
            return render_template("home.html", response_=response_)
        elif request.json:
            response_ = predict_value(request.json)
            return jsonify(response_)

if __name__ == "__main__":
    app.run(debug=True)