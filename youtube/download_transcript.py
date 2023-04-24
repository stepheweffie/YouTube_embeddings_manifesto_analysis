from youtube_transcript_api import YouTubeTranscriptApi

# Return the transcript as a list of dicts


def download_transcript(video_id, language_code):
    transcript_data = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=[language_code])
    return transcript_data

