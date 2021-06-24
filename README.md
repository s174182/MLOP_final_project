MLOP_final_project
==============================
This project is based on object detection for detecting sheeps in JPG images.

It is based on the faster rcnn network. 
For doing the transfer learning a data set from kaggle can be downloaded by running the make_dataset.py file which can be found in src->data.
    The file will also do some data manipulation and augment the dataset using Kornia. The data will be stored in data->proccesed and data->raw.
    

Transfer learning is applied by running train_model.py, which is found in src->models
    The model will take the input whether you want to train the model on augmented data or "normal" (unaugmented data). And then save the model under models.
    Due to github max limitations of 100mb one cannot save the models here but only in cloud or locally.
    
Finally inference can be made by runnning the inference.py. In here you can specify which model that shall be used for inference, (vanilla,normal or augmented).
    Vanilla model is the pretrained faster rcnn where no transfer learning has been applied to.

Inference can also be made using a pretrained model which is deployed to azure. This can be accessed running inference_endpoint.py which is found at src/models.
    The inference_endpoint takes in an image and the wanted threshold. It then returns the image with bounding boxes fulfilling the threshold requirement. 
    
Sheep counting!


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
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
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
