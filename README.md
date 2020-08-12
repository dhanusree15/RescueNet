# Disaster Response

### Instructions:
1. Navigate to the project's root directory in the terminal

2. Run the following command to install the project requirements:
    `pip install -r requirements.txt`

3. Run the following commands to set up the database and model:

    - To run the ETL pipeline that cleans and stores the data:
        `python data/process_data.py data/disaster_messages.csv data/
        disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains and saves the classifier:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Navigate to the project's `app/` directory in the terminal

5. Run the following command to run the web app:
    `python run.py`

6. Go to http://0.0.0.0:3001/ in the browser
