import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('data.pickle', 'rb'))

# for i, sample in enumerate(data_dict['data']):
#     print(f"Sample {i} length: {len(sample)}")

max_len = 42  # or compute max with max(len(s) for s in data_dict['data'])
padded_data = []

for sample in data_dict['data']:
    sample = list(sample)
    if len(sample) < max_len:
        sample += [0] * (max_len - len(sample))
    else:
        sample = sample[:max_len]
    padded_data.append(sample)

data = np.asarray(padded_data)

# data = np.asarray(data_dict['data'])
# data = np.array(data_dict['data'], dtype=object)  # but sklearn won’t accept this later
labels = np.asarray(data_dict['labels'])
print('Data shape: {}'.format(data.shape))
print('Labels shape: {}'.format(labels.shape))
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
