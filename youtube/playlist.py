import json
import pandas as pd
import os
import pickle
from download_transcript import download_transcript
import shutil
from youtube_transcript_api._errors import TranscriptsDisabled
# Opening JSON file


def get_playlist():
    # Get the Maps of Meaning playlist in a dataframe
    playlist = pd.DataFrame()
    with open('playlist.json') as json_file:
        data = json.load(json_file)
        for col in data:
            playlist[col] = data[col]
        playlist = playlist['items']
        row = {}
        video_ids = []
        for i in range(len(playlist)):
            row[i] = playlist[i]['snippet']['resourceId']['videoId']
            video_ids.append(row[i])
        all_transcripts = os.listdir('data/@JordanBPeterson')
        for video_id in video_ids:
            if video_id + '.pkl' not in all_transcripts:
                # Get the transcript
                try:
                    transcript = download_transcript(video_id, 'en')
                except TranscriptsDisabled:
                    continue
                except FileNotFoundError:
                    continue
                # Save the transcript
                with open(f"data/@JordanBPeterson/{video_id}.pkl", "wb") as f:
                    pickle.dump(transcript, f)
        os.chdir('data/@JordanBPeterson')
        os.mkdir('playlist')
        for video_id in video_ids:
            try:
                shutil.move(f'{video_id}.pkl', 'playlist')
            except FileNotFoundError:
                continue
        shutil.move('playlist', '../')
        os.chdir('../')


get_playlist()

