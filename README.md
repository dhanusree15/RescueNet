

# RescueNet
=======
# Disaster Response Pipeline Project
=======
# Disaster Response



### Instructions:
1. Navigate to the project's root directory in the terminal
=======


#### Data
The [data](https://appen.com/datasets/combined-disaster-response-data/) contains 26,248 messages that were received during past disasters around the world, such as a 2010 earthquake in Haiti and a 2012 super-storm (Sandy) in the U.S.
Each message is classified as 1 or more of the following 36 categories: <br />
'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'

None of the messages in the dataset were categorized as `child_alone` so this category was removed altogether before building the classifier, leaving 35 categories to classify.

#### Results

This application uses a logistic regression model to classify these 35 categories.
The model was evaluated on a test dataset with the following results:

- Average accuracy: 0.9483 <br />
- Average precision: 0.9397 <br />
- Average recall: 0.9483 <br />
- Average F-score: 0.9380

## Getting Started

#### Setup

1. Make sure Python 3 is installed

2. Navigate to the project's root directory in the terminal

3. Run the following command to install the project requirements:
    `pip install -r requirements.txt`

4. Run the following commands to set up the database and model:

    - To run the ETL pipeline that cleans and stores the data:
        `python data/process_data.py data/disaster_messages.csv data/
        disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains and saves the classifier:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

5. Navigate to the project's `app/` directory in the terminal

6. Run the following command to run the web app:
    `python run.py`

=======

7. Navigate to http://0.0.0.0:3001/ in the browser

#### Files

- `app/`
    - `run.py`
    - `templates/`
        - `index.html`
        - `result.html`
        
- `data/`
    - `process_data.py`
    - `disaster_messages.csv`
    - `disaster_categories.csv`

- `models/`
    - `train_classifier.py`
    
- `notebooks/`
    - `etl_pipeline.ipynb`
    - `ml_pipeline.ipynb`
    - `dashboard_visuals.ipynb`
    
- `requirements.txt`


## License
This repository is licensed under a [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).

