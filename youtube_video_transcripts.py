import pickle
import argparse
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
from transcript_data import download_transcript
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, NoTranscriptAvailable
import scrapetube

load_dotenv()

api_key = os.environ['YOUTUBE_API_KEY']
youtube = build("youtube", "v3", developerKey=api_key)


def get_channel_id(channel_url):
    channel_id = channel_url.split("/")[-1]
    return channel_id


def get_video_ids(channel_id):
    video_ids = []
    videos = scrapetube.get_channel(channel_id)
    for video in videos:
        print(video['videoId'])
        video_ids.append(video['videoId'])
    return video_ids


# Get the last video to update the data


def get_latest_video(channel_id):
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        order="date",
        maxResults=1,
        type="video",
    )
    response = request.execute()
    if response["items"]:
        video = response["items"][0]
        return video["id"]["videoId"], video["snippet"]["title"]
    return None


def process_video_transcripts(channel_url):
    channel_id = get_channel_id(channel_url)
    video_ids = get_video_ids(channel_id)
    videos = {}
    try:
        for video_id in video_ids:
            try:
                download = download_transcript(video_id=video_id, language_code='en')
                videos[video_id] = download
                if not os.path.exists(f'data/{channel_name}/{video_id}.pkl'):
                    with open(f'data/{channel_name}/{video_id}.pkl', 'wb') as pkl:
                        pickle.dump(download, pkl, protocol=pickle.HIGHEST_PROTOCOL)
            except TranscriptsDisabled:
                pass
            except NoTranscriptAvailable:
                pass
            except NoTranscriptFound:
                pass
    except RuntimeError:
        print("RuntimeError")


def main(channel_url):
    process_video_transcripts(channel_url)
    # get_video_ids(get_channel_id(channel_url))
    # get_latest_video(get_channel_id(channel_url))


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process transcripts for a YouTube channel')
    parser.add_argument('channel_name', type=str, help='The YouTube channel name')
    parser.add_argument('channel_url', type=str, help='The URL of the YouTube channel')
    args = parser.parse_args()
    channel_name = args.channel_name
    if not os.path.exists(f'data/{channel_name}'):
        os.makedirs(f'data/{channel_name}')
    main(args.channel_url)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
