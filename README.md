University Rating
==============================
Finding chance of getting admission in university

Project Organization
------------

    ├── README.md          <- Documentation of the project
    ├── data
    │   ├── processed      <- Processed data
    │   └── raw            <- Input data
    │
    ├── models             <- Saved models, scalers and other saved states
    │
    ├── notebooks          <- Experimenting with data
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── tests               <- testing files
    │   ├── test_config.py  <- Tests for inputs and outputs
    │   │ 
    │   └── conftest.py     <- Fixtures for tests
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── predict_model.py
    │       └── train_model.py
    │   
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------



1. Add the storage files in data_given/*.csv</br>
2. Run the program using dvc repro
