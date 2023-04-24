import json
from youtube_transcript_api import YouTubeTranscriptApi
import d6tflow
import pickle
import os
from dotenv import load_dotenv
from youtube_transcript_api.formatters import JSONFormatter
load_dotenv()
# List of Video IDs
video = [0]
# The Video ID For A Single Video Stored In .env
stored_video = os.environ['YOUTUBE_VIDEO_ID']

# Return the transcript as a list of dicts


def download_transcript(video_id, language_code):
    transcript_data = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=[language_code])
    return transcript_data

# Format transcript to JSON runs as transcript = download_transcript(video_id, language_code) format_to_json(transcript)


def format_to_json(transcript):
    formatter = JSONFormatter()
    json_formatted = formatter.format_transcript(transcript)
    with open('transcript.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_formatted)

# Make the call


class DownloadTranscriptTask(d6tflow.tasks.TaskPickle):
    video_id = ''
    language_code = 'en'
    if len(video) == 0:
        video_id = stored_video

    def output(self):
        if self.video_id == stored_video:
            return d6tflow.targets.PickleTarget('single_video_data.pkl')
        else:
            return d6tflow.targets.PickleTarget('more_videos_data.pkl')

    def run(self):
        if type(self.video_id) is list:
            transcripts_pkl = {}
            for vid in self.video_id:
                self.video_id = vid
                transcripts_pkl[vid] = download_transcript(self.video_id, self.language_code)
            self.save(transcripts_pkl)
        else:
            transcript_pkl = download_transcript(self.video_id, self.language_code)
            self.save(transcript_pkl)

# Unpickle the data to video.json


class LoadDataTask(d6tflow.tasks.TaskJson):
    def output(self):
        return d6tflow.targets.JsonTarget('video.json')

    def run(self):
        file = handle_pickle()
        with open("video.json", 'w') as f:
            json.loads(file)
            print(f)
            f.write(file)
            f.close()

# Pickle helper function


def handle_pickle():
    file = input('Give me a pickle to dump to video.json: ')
    with open(file, 'rb') as f:
        data = pickle.load(f)
        json_string = json.dumps(data)
        f.close()
    return json_string


if __name__ == "__main__":
    LoadDataTask().run()
