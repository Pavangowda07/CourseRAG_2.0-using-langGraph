import os
import re
import json
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import yt_dlp
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})", url)
    if not match:
        raise ValueError(f"Invalid YouTube URL: {url}")
    return match.group(1)

def format_timestamp(seconds: int) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

# --- PATH A: YouTube Transcript API ---
def fetch_youtube_segments(video_url: str) -> list[dict] | None:
    try:
        video_id = extract_video_id(video_url)
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["en", "hi"])
        
        segments = []
        for snippet in fetched:
            segments.append({
                "start_sec": int(snippet.start),
                "timestamp": format_timestamp(snippet.start),
                "text": snippet.text.strip()
            })
        return segments
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception as e:
        print(f"[YouTube API Warning]: {e}")
        return None

# --- PATH B: Audio Download + Gemini Timestamped Transcription ---
def transcribe_audio_segments(video_url: str) -> list[dict]:
    """Downloads audio from a video URL, transcribes it using Gemini, and returns timestamped segments."""
    
    # Define temporary local filename and remove any leftover file from previous runs
    output_base = "temp_audio"
    audio_path = f"{output_base}.mp3"
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # Configure yt-dlp to download the best audio and convert it to a lightweight 64kbps MP3
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_base}.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "64",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    # Download and extract the audio locally
    print("Downloading audio via yt-dlp...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    uploaded_file = None
    try:
        # Upload the local MP3 file to the Gemini File API
        print("Uploading audio to Gemini...")
        uploaded_file = client.files.upload(file=audio_path)

        # Prompt Gemini to transcribe the audio into 15-30s chunks formatted strictly as JSON
        prompt = (
            "Transcribe the spoken audio accurately into chronological segments (roughly 15-30 seconds per segment). "
            "Return ONLY a valid JSON array where each item has 'start_sec' (integer seconds from start) "
            "and 'text' (spoken words in that segment)."
        )

        # Generate the transcript using the uploaded audio and force JSON output
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        # Parse the JSON string from Gemini and format each segment with readable timestamps
        raw_segments = json.loads(response.text)
        segments = []
        for item in raw_segments:
            sec = int(item.get("start_sec", 0))
            segments.append({
                "start_sec": sec,
                "timestamp": format_timestamp(sec),
                "text": item.get("text", "").strip()
            })
        return segments

    finally:
        # Always clean up both the local audio file and the remote Gemini file, even if an error occurs
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if uploaded_file:
            client.files.delete(name=uploaded_file.name)


# --- QUICK TEST ---
if __name__ == "__main__":
    url = input("Enter a YouTube URL to test extraction: ").strip()
    
    print("\n1. Trying YouTube Transcript API...")
    segs = fetch_youtube_segments(url)
    
    if segs:
        print(f"Success via YouTube API! Total segments: {len(segs)}")
    else:
        print("Captions not found. Running Audio + Gemini fallback...")
        segs = transcribe_audio_segments(url)
        print(f"Success via Gemini Audio! Total segments: {len(segs)}")

    print("\nFirst 3 standardized segments:")
    for s in segs[:3]:
        print(s)