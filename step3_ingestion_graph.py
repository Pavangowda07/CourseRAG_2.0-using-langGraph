import os
import re
import json
from typing import TypedDict, Optional, Any
from dotenv import load_dotenv
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from pydantic import BaseModel, Field
import yt_dlp
from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from your .env file (GOOGLE_API_KEY, LANGSMITH_API_KEY, etc.)
load_dotenv()

# Initialize the Google GenAI client (automatically reads GOOGLE_API_KEY from environment)
genai_client = genai.Client()

# ==============================================================================
# 1. HELPER FUNCTIONS (Instrumented with @traceable for LangSmith Observability)
# ==============================================================================

def extract_video_id(url: str) -> str:
    """
    Extracts the unique 11-character YouTube video ID from any standard YouTube URL
    (e.g., https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID).
    """
    match = re.search(r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})", url)
    if not match:
        raise ValueError(f"Invalid YouTube URL: {url}")
    return match.group(1)


def format_timestamp(seconds: int) -> str:
    """
    Converts raw seconds (e.g., 135) into a human-readable timestamp string (e.g., '02:15').
    Includes hours (HH:MM:SS) if the video is 1 hour or longer.
    """
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


# @traceable logs this function's inputs, outputs, and execution time as a child step in LangSmith
@traceable(name="fetch_youtube_captions", run_type="tool")
def fetch_youtube_segments(video_url: str) -> list[dict] | None:
    """
    PATH A (Primary): Tries to fetch existing captions directly from YouTube.
    Returns a standardized list of dicts: [{"start_sec": 12, "timestamp": "00:12", "text": "..."}]
    Returns None if captions are disabled, unavailable, or missing.
    """
    try:
        video_id = extract_video_id(video_url)
        ytt_api = YouTubeTranscriptApi()

        # Attempt to fetch English ('en') captions first; fall back to Hindi ('hi') if unavailable
        fetched = ytt_api.fetch(video_id, languages=["en", "hi"])

        # Standardize every caption snippet into a clean dictionary format
        return [
            {
                "start_sec": int(snippet.start),
                "timestamp": format_timestamp(snippet.start),
                "text": snippet.text.strip(),
            }
            for snippet in fetched
        ]
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        # Expected exceptions when a video has no captions -> triggers our fallback branch
        return None
    except Exception as e:
        print(f"[YouTube API Warning]: {e}")
        return None


@traceable(name="download_youtube_audio", run_type="tool")
def download_audio_file(video_url: str, output_base: str = "temp_audio") -> str:
    """
    PATH B - Step 1 (Fallback): Downloads ONLY the audio stream from YouTube using yt-dlp
    and converts it to a lightweight 64kbps MP3 file via FFmpeg.
    """
    audio_path = f"{output_base}.mp3"

    # Delete any leftover temp file from a previous interrupted run
    if os.path.exists(audio_path):
        os.remove(audio_path)

    ydl_opts = {
        "format": "bestaudio/best",          # Download only audio (skips heavy video streams)
        "outtmpl": f"{output_base}.%(ext)s", # Temporary filename pattern before conversion
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio", # Uses FFmpeg installed on your OS
                "preferredcodec": "mp3",     # Convert to .mp3
                "preferredquality": "64",    # 64kbps keeps file size small & upload fast
            }
        ],
        "quiet": True,                       # Suppress verbose yt-dlp terminal logs
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return audio_path


# Define the exact schema Gemini must follow
class TranscriptSegment(BaseModel):
    start_sec: int = Field(description="Start time of this segment in integer seconds from the beginning of the audio.")
    text: str = Field(description="The spoken words transcribed in this segment.")

# ************************************************************************
# AUDIO TO TEXT 


@traceable(name="gemini_audio_transcribe", run_type="llm")
def transcribe_audio_with_gemini(audio_path: str) -> list[dict]:
    """
    PATH B - Step 2 (Fallback): Uploads the downloaded MP3 to Google AI Studio's File API,
    enforces a strict Pydantic schema via response_schema, and cleans up temporary files.
    """
    uploaded_file = None
    try:
        # 1. Upload the local MP3 file to Gemini's servers
        uploaded_file = genai_client.files.upload(file=audio_path)

        # 2. Prompt Gemini (schema rules are handled automatically by response_schema)
        prompt = (
            "Transcribe the spoken audio accurately into chronological segments "
            "(roughly 15-30 seconds per segment)."
        )

        # 3. Enforce the Pydantic list[TranscriptSegment] schema
        response = genai_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[TranscriptSegment],
            ),
        )

        # 4. response.parsed is already a validated list[TranscriptSegment]
        validated_segments: list[TranscriptSegment] = response.parsed
        return [
            {
                "start_sec": seg.start_sec,
                "timestamp": format_timestamp(seg.start_sec),
                "text": seg.text.strip(),
            }
            for seg in validated_segments
        ]
    finally:
        # Always delete the local MP3 and cloud file, even if an error occurs mid-run
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if uploaded_file:
            genai_client.files.delete(name=uploaded_file.name)



@traceable(name="chunk_and_embed_faiss", run_type="retriever")
def create_faiss_vectorstore(
    segments: list[dict], video_id: str, max_chars: int = 600
) -> FAISS:
    """
    CONVERGENCE STEP: Combines small caption segments into ~600-character chunks
    while preserving the starting timestamp in each chunk's metadata, then embeds
    them into an in-memory FAISS vector database.
    """
    documents = []
    current_texts = []
    current_length = 0
    chunk_start_sec = 0
    chunk_start_ts = "00:00"

    for i, seg in enumerate(segments):
        # Record the timestamp of the very first segment in this new chunk
        if not current_texts:
            chunk_start_sec = seg["start_sec"]
            chunk_start_ts = seg["timestamp"]

        current_texts.append(seg["text"])
        current_length += len(seg["text"]) + 1

        # Once the accumulated text reaches ~600 chars (or we hit the last segment), finalize chunk
        if current_length >= max_chars or i == len(segments) - 1:
            combined_text = " ".join(current_texts)

            # Wrap text + timestamp metadata into a LangChain Document object
            documents.append(
                Document(
                    page_content=f"[{chunk_start_ts}] {combined_text}",
                    metadata={
                        "video_id": video_id,
                        "start_sec": chunk_start_sec,
                        "timestamp": chunk_start_ts,
                        "youtube_link": f"https://www.youtube.com/watch?v={video_id}&t={chunk_start_sec}s",
                    },
                )
            )

            # Overlap: Carry the last segment into the next chunk so sentences split
            # across boundaries remain searchable
            if i < len(segments) - 1:
                current_texts = [seg["text"]]
                current_length = len(seg["text"])
                chunk_start_sec = seg["start_sec"]
                chunk_start_ts = seg["timestamp"]
            else:
                current_texts = []

    # Convert every Document chunk into a vector embedding using Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return FAISS.from_documents(documents, embeddings)


# ==============================================================================
# 2. LANGGRAPH STATE & NODES (Defines the Graph Memory and Processing Steps)
# ==============================================================================

class IngestionState(TypedDict):
    """
    The shared state dictionary passed between every node in the LangGraph workflow.
    Each node reads from this dictionary and returns a dict with keys to update.
    """
    video_url: str
    video_id: Optional[str]
    segments: Optional[list[dict]]
    source: Optional[str]          # Tracks whether data came from "youtube_api" or "gemini_audio"
    vectorstore: Optional[Any]     # Stores the compiled FAISS vector index


def node_try_youtube_api(state: IngestionState) -> dict:
    """Node 1: Extracts the video ID and tries to pull native YouTube captions."""
    url = state["video_url"]
    vid_id = extract_video_id(url)
    print(f"\n[Node 1] Checking YouTube Transcript API for ID: {vid_id}...")

    segments = fetch_youtube_segments(url)
    if segments:
        print(f" -> Found {len(segments)} native caption segments.")
        return {"video_id": vid_id, "segments": segments, "source": "youtube_api"}

    print(" -> Native captions not found.")
    return {"video_id": vid_id, "segments": None}


def node_audio_fallback(state: IngestionState) -> dict:
    """Node 2 (Fallback): Downloads audio via yt-dlp and transcribes it via Gemini."""
    print("[Node 2 - Fallback] Downloading audio & transcribing with Gemini...")
    audio_path = download_audio_file(state["video_url"])
    segments = transcribe_audio_with_gemini(audio_path)
    print(f" -> Generated {len(segments)} timestamped segments via Gemini.")
    return {"segments": segments, "source": "gemini_audio"}


def node_build_vectorstore(state: IngestionState) -> dict:
    """Node 3 (Convergence): Builds the timestamp-aware FAISS vector store from segments."""
    print("[Node 3 - Convergence] Chunking & building FAISS index...")
    vectorstore = create_faiss_vectorstore(
        segments=state["segments"],
        video_id=state["video_id"],
        max_chars=600,
    )
    print(" -> FAISS vector store ready.")
    return {"vectorstore": vectorstore}


def route_transcript(state: IngestionState) -> str:
    """
    Conditional Edge Router ("Transcript Found?" diamond in your diagram).
    Returns 'found' if segments exist, or 'missing' to trigger the audio fallback.
    """
    if state.get("segments"):
        print("Decision -> YES (Routing directly to vector store)")
        return "found"
    print("Decision -> NO (Routing to audio fallback)")
    return "missing"


# ==============================================================================
# 3. COMPILE PHASE 1 INGESTION GRAPH
# ==============================================================================

def build_ingestion_graph():
    """Assembles all nodes and conditional edges into a runnable LangGraph workflow."""
    builder = StateGraph(IngestionState)

    # Register the 3 nodes in the graph
    builder.add_node("try_youtube_api", node_try_youtube_api)
    builder.add_node("audio_fallback", node_audio_fallback)
    builder.add_node("build_vectorstore", node_build_vectorstore)

    # Start -> Node 1
    builder.add_edge(START, "try_youtube_api")

    # Node 1 -> Conditional Branch ("found" skips fallback; "missing" runs fallback)
    builder.add_conditional_edges(
        "try_youtube_api",
        route_transcript,
        {
            "found": "build_vectorstore",
            "missing": "audio_fallback",
        },
    )

    # Node 2 (Fallback) -> Converges back into Node 3
    builder.add_edge("audio_fallback", "build_vectorstore")

    # Node 3 -> End of Phase 1
    builder.add_edge("build_vectorstore", END)

    return builder.compile()


# Compile the graph once at module level so it can be imported by Streamlit later
ingestion_app = build_ingestion_graph()


# ==============================================================================
# 4. LOCAL EXECUTION & LANGSMITH TRACE TEST
# ==============================================================================
if __name__ == "__main__":
    test_url = input("Enter a YouTube URL to ingest: ").strip()

    # Run the graph; 'run_name' sets the top-level trace name in your LangSmith dashboard
    result = ingestion_app.invoke(
        {"video_url": test_url},
        config={"run_name": "YouTube_Ingestion_Pipeline"},
    )

    print("\n=== Ingestion Complete ===")
    print(f"Video ID : {result['video_id']}")
    print(f"Source   : {result['source']}")
    print("Check your 'youtube-timestamp-rag' project in LangSmith to view the trace tree!")