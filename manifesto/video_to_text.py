import pandas as pd

data = pd.read_pickle(f'single_video_transcript.pkl')
data = pd.DataFrame(data)
data = data['text']
data = data.apply(lambda x: x.replace('\n', ' '))
data = data.apply(lambda x: x.replace('.', '\n'))
data = data.apply(lambda x: x.replace('?', '\n'))
data_string = ''
for i in range(len(data)):
    data_string += data.iloc[i]
print(data_string)
with open('single_video.txt', 'w') as f:
    f.write(data_string)



