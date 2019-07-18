import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

essays = pd.read_csv('../../data/processed/essays_with_topic_scores.csv', index_col=0)

# Set the essay id as the index of the dataframe
essays.set_index('essay_id', inplace=True)

# scale the length column so that all features are between 0 - 1
mm_scaler = preprocessing.MinMaxScaler()
essays['length'] = mm_scaler.fit_transform(essays[['length']])

scores = essays['domain1_score']
length = essays['length']

# Linear correlated data
plt.scatter(length, scores, color='g')
plt.xlabel('Essay Length')
plt.ylabel('Essay Score')
plt.show()

topic1 = essays['bad_if_kids_spend_too_much_time']
topic2 = essays['doesnt_have_the_negative_exercise_effect']
topic3 = essays['games_and_information']
topic4 = essays['looking_at_websites_for_info']
topic5 = essays['spend_time_looking_on_websites']

# all the other topics plotted against essay score
fig = plt.figure(figsize=(15,10))

plt.scatter(topic1, scores, color='b', label='Topic 1')
plt.scatter(topic2, scores, color='r', label='Topic 2')
plt.scatter(topic3, scores, color='g', label='Topic 3')
plt.scatter(topic4, scores, color='y', label='Topic 4')
plt.scatter(topic5, scores, color='k', label='Topic 5')

plt.xlabel('Topics')
plt.ylabel('Essay Score')
plt.legend(loc='lower center', )
plt.xticks(np.arange(0, 1.05, 0.05))
plt.yticks(np.arange(0, 12.5, 0.5))
plt.show()