import numpy as np
from bertopic import BERTopic
from scipy import stats

from src.utils import *

percentile = 0.9
df = load_df(percentile=percentile)

topic_model = BERTopic(min_topic_size=100)
topics, probs = topic_model.fit_transform(df['explanation'])

topic_model.visualize_topics().show()
print(topic_model.get_topic_info())
print(f'number of unassigned explanations: {len([topic for topic in topics if topic == -1])})')

groups = [df.loc[np.array(topics) == i]['layer'].array for i in range(len(topic_model.get_topic_info()))]
f_statistic, p_value = stats.f_oneway(*groups)

print(f"{f_statistic=},{p_value=}")
print("done!")