def uniqueColumns(df):
    for column in df.columns:
        print(column)

def printEssaySetStats(df):
    print('Essay Set #{0} Length of dataset {1}'.format(df.essay_set.unique()[0], len(df)))
    print(df.isnull().sum(axis=0))
    print('\n')