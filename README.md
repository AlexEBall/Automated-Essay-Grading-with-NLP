# Automated-Essay-Grading-with-NLP

## Project Structure
```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks.
|   └── RFClassifier_new_data.ipynb
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_phrase_models.py
|   |   └── create_essay_with_topics_df_for_ml.py
|   |   └── data_preprocessing.py
|   |   └── grammatical_features.py
|   |   └── topic_modeling_with_LDA.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make predictions
│   │   │                 
│   │   ├── random_forest.py
│   │   └── random_search_random_forest_classifier.py
|   |   └── sgd_linear_application.py
|   |   └── svm_train_and_predict.py
|   |
|   ├── to-dos         <- Things I'dl like to do in the future
│   │   └── snippets.py
|   |
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── essay_EDA.py
│       └── essay_set_1_LDA_visualiztion.py
|       └── essay_set_1_model_visualiztion.py
|       └── LDA_visualization_explained.md
└── utils           
    └── helpers.py
```

### Purpose of project
For this project I wanted to see if I could create a model that would aid teacher's with grading student essays.
I have two trained models one whose performance is quite bad and the other whose performace is quite good. It's interesing to compare and contrast the reasons behind that. In the future I could perhaps take the best from both models and see what comes of that.

### The dataset
The original dataset can be obtained at this [link](https://www.kaggle.com/c/asap-aes/data) and is provided by the Hewlett Foundation in cooperation with Kaggle

### Google slides presentation
Here you can view a google slides presentaiton that goes more in-depth of the problem at hand and the lessons I learned

#### Contributing
If you'd like to lend a hand or have any suggestions to make the code more performant I'm always open to colaboration. Please either email me or make a pull request. 