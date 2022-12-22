from src.models.predict_model import predict_value

data = {
    "GRE Score": 300,
    "CGPA": 9.98,
    "University Rating": 4,
    "SOP": 4,
    "LOR": 4,
    "Research": 1

}
def test_input(retrieve_schema):
    for i in data.keys():
        if "min" in retrieve_schema[i].keys():
            assert retrieve_schema[i]["min"] <data[i] <= retrieve_schema[i]["max"]
        else:
            assert data[i] in retrieve_schema[i]["values"]

def test_output(retrieve_schema):
    response_ = predict_value(data)
    assert retrieve_schema["output"]["min"] < response_ <= retrieve_schema["output"]["max"]
