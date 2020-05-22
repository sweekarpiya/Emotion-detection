# Emotion-detection
Emotion-detection (NLP) project assigned at workflow training.

## Project Structure
```
├── README.md          <- README file.
├── api                <- APIs to interact with the inference model.
│   ├── configs        	<- Config for API
│   ├── controllers    	<- Controller logic for API routes
│   ├── routes         	<- Rules for end-points
│   ├── static         	<- Static files for template engine
│   ├── templates      	<- templates to be rendered
│   ├── app.py         	<- Main API file
│   ├── settings.py    	<- Settings file for Database
|
├── checkpoints        	<- Checkpoints/saves of trained Machine Learning model.
|
├── docs               <- Project related analysis and other documents
├── env                <- Virtual environments
├── logs               <- Logs of the project
├── ml                 <- ML related files
	├── data           <- Dataset for training
	    └── external   <- Secondary data sources.
	├── models             <- Trained and 		  serialized models/artifacts
	├── src             <- Source code for training prediction
	│   ├── __init__.py
	│   │
	│   ├── configs         <- Contains the config files.
	│   │   │
	│   │   └── config.py
	|   |
	│   ├── data           <- Scripts to download data and store on root data path.
	│   │   │
	│   │   └── get_dataset.py
	|   |
	│   ├── evaluate     <- Scripts for evaluation of trained model
	│   │   │
	│   │   └── evaluation.py
	│   │
	│   ├── features       <- Scripts to process the data.
	│   │   │
	│   │   ├── build_features.py
	│   │   └── text_preprocessor.py
	│   │
	│   ├── models         <- Scripts to train model, and predict emotion
	│   │   │
	│   │   ├── predict_model.py
	│   │   └── train_model.py
	|   |
	│   ├── utils          <- Collection of various utility functions.
	│   │   │
	|   |   ├── data_augment.py
	|   |   ├── dispatcher.py
	|   |   ├── lower_case.py
	|   |   ├── symbol_removal.py
	|   |   └── tokenizer.py
	|   |
	│   ├── main.py      <- script to run the flask web app/ train model
|	|   |
├── notebooks          <- Data analysis Jupyter notebooks
├── out          <- Output reports of training
│
├── requirements.txt   <- Pip generated requirements file for the project.
├── Dockerfile		   <- Docker file to generate build image.
	
```

## Getting Started

Generate a virtual environment to avoid any conflicts.

```
python3 -m venv /env/emotion
source /env/emotion/bin/activate

```

### Requirements

```
pip install -r requirements.txt
```

### Configuration
Highly recommended to check the config files of both API and ML. They contain information of constants used throughout the source code.

### Run

#### Training model and starting API Server

First run of the project requires you to train the model, as the checkpoints aren't pushed to github, due to heavy memory space.

Run Command
```
python3 ml/src/main.py --train <train_option> --model <model_name> --save <y or n> --feature <feature_method>
```
mlflow logging at 5000 port

- ```train_option -> yes/no```
- ```model_name -> naive_bayes, svc, softmax_l1, softmax_l2, rand_clf```
- ```save -> y/n (save classification report in csv and heatmap as png or not)```
- ```feature_method -> count, tfidf```

__NOTE__: Use ```yes, rand_clf, n, count``` for training the model on the first run.

Opens Flask server at port 5555

#### API Server
```
python3 ml/src/main.py
```
#### End-points

- ```/``` - Initial API page(Template to redirect to create/prediction page)
- ```/api/v1/create``` - [GET] Form to input sentence for classification (Template)
- ```/api/v1/create``` - [POST] Renders the emotion prediction of the sentence (Template)
- ```/api/v1/predictions``` - [GET] Sends JSON response of the stored prediction documents (API)
- ```/api/v1/predictions/<emotion>``` - [GET] Sends JSON response of the stored prediction documents given the emotion. (API)


### Reproducibility
Before docker ensure you have trained the model.

#### Run using docker-compose file
```
sudo docker-compose up
```
API runs on ```127.0.0.1:5555```
