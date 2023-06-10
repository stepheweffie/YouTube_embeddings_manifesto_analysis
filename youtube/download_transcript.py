from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import os
from googleapiclient.discovery import build
import json
load_dotenv()
channel_name = '@JordanBPeterson'

api_key = os.environ['YOUTUBE_API_KEY']
youtube = build("youtube", "v3", developerKey=api_key)
# Return the transcript as a list of dicts


def download_transcript(video_id, language_code):
    transcript_data = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=[language_code])
    return transcript_data


def get_playlist(playlist_id):
    playlist = youtube.playlistItems().list(
        part="snippet",
        playlistId=playlist_id,
        maxResults=50,
    )
    response = playlist.execute()
    with open("playlist.json", "w") as outfile:
        json.dump(response, outfile, indent=4)
    return response


# Get the Maps of Meaning playlist
get_playlist(playlist_id='PL22J3VaeABQCn5nTAx65NRlh1EsKD0UQD')