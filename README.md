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

###