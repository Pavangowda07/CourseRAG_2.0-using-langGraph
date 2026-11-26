import os
import re
from typing import TypedDict, Optional, Any, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import yt_dlp
from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# Load API keys (GOOGLE_API_KEY, LANGSMITH_API_KEY, LANGSMITH_PROJECT) from .env
load_dotenv()

# Initialize Google GenAI SDK client (used for file uploads & structured audio transcription)
genai_client = genai.Client()


# ==============================================================================
# 1. PYDANTIC SCHEMAS (Enforce strict JSON structures from Gemini LLM responses)
# ==============================================================================

class TranscriptSegment(BaseModel):
    """Represents a single time-stamped chunk of English transcript text."""
    start_sec: int = Field(
        description="Start time of this segment in integer seconds from the beginning."
    )
    english_text: str = Field(
        description="Spoken content transcribed and translated accurately into English."
    )


class AudioTranscriptionResult(BaseModel):
    """Schema returned when Gemini transcribes and translates raw audio in Path B."""
    detected_language: str = Field(
        description="The primary spoken language detected in the audio (e.g., 'English', 'Hindi', 'Kannada')."
    )
    segments: list[TranscriptSegment] = Field(
        description="Chronological English transcript segments with start timestamps."
    )


class QueryAnalysis(BaseModel):
    """Classifies user intent and resolves pronouns/references using prior chat history."""
    intent: Literal["summary", "qa"] = Field(
        description="Choose 'summary' if the user asks for an overview, key takeaways, or full summary of the video; choose 'qa' for specific questions or follow-ups."
    )
    standalone_query: str = Field(
        description="Rewrite the user's message into a clear standalone search query using chat history context (if needed)."
    )


class TimestampCitation(BaseModel):
    """Represents a single verified timestamp milestone in the video."""
    timestamp: str = Field(description="Formatted timestamp like '04:15'.")
    start_sec: int = Field(description="Exact integer start seconds for clickable URL links.")
    topic_note: str = Field(description="Brief description of what was covered at this timestamp.")


class AssistantResponse(BaseModel):
    """Unified structured response returned for summaries, specific Q&A, and follow-up chats."""
    found_in_video: bool = Field(
        description="True if the requested topic/summary is supported by the video content; False if the topic is not found in the video."
    )
    response_type: Literal["summary", "qa"] = Field(
        description="Indicates whether this response is a full video 'summary' or a specific 'qa' answer."
    )
    answer: str = Field(
        description="Comprehensive, well-formatted English response grounded strictly in the video content. If found_in_video is False, state clearly that the topic is not found in this video."
    )
    citations: list[TimestampCitation] = Field(
        default_factory=list,
        description="Key timestamps supporting the answer or summary milestones (empty if found_in_video is False)."
    )


# ==============================================================================
# 2. HELPER FUNCTIONS (Instrumented with @traceable for LangSmith Observability)
# ==============================================================================

def extract_video_id(url: str) -> str:
    """Extracts the 11-character YouTube video ID from any standard YouTube link."""
    match = re.search(r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})", url)
    if not match:
        raise ValueError(f"Invalid YouTube URL: {url}")
    return match.group(1)


def format_timestamp(seconds: int) -> str:
    """Converts integer seconds into a readable timestamp string ('MM:SS' or 'HH:MM:SS')."""
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


@traceable(name="translate_segments_to_english", run_type="llm")
def translate_segments_with_gemini(raw_segments: list[dict], source_lang: str) -> list[dict]:
    """Backup translator when YouTube captions exist in another language without auto-translation."""
    prompt = (
        f"Translate the following video caption segments from {source_lang} into clear, accurate English. "
        "Keep the exact same 'start_sec' values and segment count."
    )
    response = genai_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[prompt, str(raw_segments)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[TranscriptSegment],
        ),
    )
    return [
        {
            "start_sec": seg.start_sec,
            "timestamp": format_timestamp(seg.start_sec),
            "text": seg.english_text.strip(),
        }
        for seg in response.parsed
    ]


@traceable(name="fetch_youtube_captions", run_type="tool")
def fetch_youtube_segments(video_url: str) -> tuple[list[dict], str] | None:
    """
    PATH A (Primary): Fetches native captions in any available language, records the
    original language, and standardizes all text into English.
    """
    try:
        video_id = extract_video_id(video_url)
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        # Prefer English first; otherwise grab the first available language track
        selected_transcript = None
        for t in transcript_list:
            if t.language_code.startswith("en"):
                selected_transcript = t
                break
        if selected_transcript is None:
            selected_transcript = next(iter(transcript_list))

        original_lang = selected_transcript.language.split(" (")[0]

        # Translate non-English captions into English while preserving timestamps
        if not selected_transcript.language_code.startswith("en"):
            if selected_transcript.is_translatable:
                fetched = selected_transcript.translate("en").fetch()
                segments = [
                    {
                        "start_sec": int(s.start),
                        "timestamp": format_timestamp(s.start),
                        "text": s.text.strip(),
                    }
                    for s in fetched
                ]
                return segments, original_lang
            else:
                raw = [{"start_sec": int(s.start), "text": s.text} for s in selected_transcript.fetch()]
                return translate_segments_with_gemini(raw, original_lang), original_lang

        # Already English
        fetched = selected_transcript.fetch()
        segments = [
            {
                "start_sec": int(s.start),
                "timestamp": format_timestamp(s.start),
                "text": s.text.strip(),
            }
            for s in fetched
        ]
        return segments, "English"

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, StopIteration):
        return None
    except Exception as e:
        print(f"[YouTube API Warning]: {e}")
        return None


@traceable(name="download_youtube_audio", run_type="tool")
def download_audio_file(video_url: str, output_base: str = "temp_audio") -> str:
    """PATH B - Step 1 (Fallback): Downloads audio-only MP3 via yt-dlp + FFmpeg."""
    audio_path = f"{output_base}.mp3"
    if os.path.exists(audio_path):
        os.remove(audio_path)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_base}.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return audio_path


@traceable(name="gemini_audio_transcribe_and_translate", run_type="llm")
def transcribe_audio_with_gemini(audio_path: str) -> tuple[list[dict], str]:
    """PATH B - Step 2 (Fallback): Detects spoken language and transcribes/translates to English."""
    uploaded_file = None
    try:
        uploaded_file = genai_client.files.upload(file=audio_path)

        prompt = (
            "Listen to this audio carefully. Identify the primary spoken language "
            "(e.g., Hindi, English, Kannada, etc.). Then transcribe the entire spoken content "
            "into chronological segments (roughly 15-30 seconds per segment). "
            "If the audio is NOT in English, translate every segment accurately into natural English "
            "while preserving the exact start_sec timestamp."
        )

        response = genai_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AudioTranscriptionResult,
            ),
        )

        parsed: AudioTranscriptionResult = response.parsed
        segments = [
            {
                "start_sec": seg.start_sec,
                "timestamp": format_timestamp(seg.start_sec),
                "text": seg.english_text.strip(),
            }
            for seg in parsed.segments
        ]
        return segments, parsed.detected_language

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if uploaded_file:
            genai_client.files.delete(name=uploaded_file.name)


@traceable(name="chunk_and_embed_faiss", run_type="retriever")
def create_chunks_and_vectorstore(
    segments: list[dict], video_id: str, max_chars: int = 600
) -> tuple[list[Document], FAISS]:
    """
    CONVERGENCE STEP: Builds timestamped Document chunks AND embeds them into FAISS.
    Returns both (documents, vectorstore) so full-video summaries can read all chunks
    chronologically while specific Q&A uses FAISS vector similarity search.
    """
    documents = []
    current_texts = []
    current_length = 0
    chunk_start_sec = 0
    chunk_start_ts = "00:00"

    for i, seg in enumerate(segments):
        if not current_texts:
            chunk_start_sec = seg["start_sec"]
            chunk_start_ts = seg["timestamp"]

        current_texts.append(seg["text"])
        current_length += len(seg["text"]) + 1

        if current_length >= max_chars or i == len(segments) - 1:
            combined_text = " ".join(current_texts)
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

            # 1-segment overlap across chunk boundaries
            if i < len(segments) - 1:
                current_texts = [seg["text"]]
                current_length = len(seg["text"])
                chunk_start_sec = seg["start_sec"]
                chunk_start_ts = seg["timestamp"]
            else:
                current_texts = []

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return documents, vectorstore


# ==============================================================================
# 3. PHASE 1: LANGGRAPH INGESTION WORKFLOW (Runs once per YouTube URL)
# ==============================================================================

class IngestionState(TypedDict):
    video_url: str
    video_id: Optional[str]
    segments: Optional[list[dict]]
    original_language: Optional[str]
    source: Optional[str]
    documents: Optional[list[Document]]   # Stored for full-video summaries
    vectorstore: Optional[Any]            # Stored for fast semantic Q&A retrieval


def node_try_youtube_api(state: IngestionState) -> dict:
    url = state["video_url"]
    vid_id = extract_video_id(url)
    print(f"\n[Phase 1 - Node 1] Checking YouTube Captions for {vid_id}...")

    result = fetch_youtube_segments(url)
    if result:
        segments, orig_lang = result
        print(f" -> Found captions (Original Language: {orig_lang} -> English).")
        return {
            "video_id": vid_id,
            "segments": segments,
            "original_language": orig_lang,
            "source": "youtube_api",
        }

    print(" -> Captions unavailable.")
    return {"video_id": vid_id, "segments": None}


def node_audio_fallback(state: IngestionState) -> dict:
    print("[Phase 1 - Node 2] Running Audio Fallback (yt-dlp + Gemini English Transcription)...")
    audio_path = download_audio_file(state["video_url"])
    segments, detected_lang = transcribe_audio_with_gemini(audio_path)
    print(f" -> Transcribed audio (Detected Language: {detected_lang} -> English).")
    return {
        "segments": segments,
        "original_language": detected_lang,
        "source": "gemini_audio",
    }


def node_build_vectorstore(state: IngestionState) -> dict:
    print("[Phase 1 - Node 3] Building timestamped chunks & FAISS index...")
    documents, vectorstore = create_chunks_and_vectorstore(
        segments=state["segments"],
        video_id=state["video_id"],
        max_chars=600,
    )
    return {"documents": documents, "vectorstore": vectorstore}


def route_transcript(state: IngestionState) -> str:
    return "found" if state.get("segments") else "missing"


def build_ingestion_graph():
    builder = StateGraph(IngestionState)
    builder.add_node("try_youtube_api", node_try_youtube_api)
    builder.add_node("audio_fallback", node_audio_fallback)
    builder.add_node("build_vectorstore", node_build_vectorstore)

    builder.add_edge(START, "try_youtube_api")
    builder.add_conditional_edges(
        "try_youtube_api",
        route_transcript,
        {"found": "build_vectorstore", "missing": "audio_fallback"},
    )
    builder.add_edge("audio_fallback", "build_vectorstore")
    builder.add_edge("build_vectorstore", END)
    return builder.compile()


ingestion_app = build_ingestion_graph()


# ==============================================================================
# 4. PHASE 2: CONVERSATIONAL Q&A + SUMMARIZATION GRAPH (Runs per User Message)
# ==============================================================================

class QAState(TypedDict):
    question: str
    chat_history: list[dict]           # List of {"role": "user"|"assistant", "content": "..."}
    video_id: str
    original_language: str
    documents: list[Document]          # Full chronological chunks (used for summaries)
    vectorstore: Any                   # FAISS index (used for targeted Q&A)
    intent: Optional[str]              # "summary" or "qa"
    standalone_query: Optional[str]    # Context-resolved query for FAISS
    context_docs: Optional[list[Document]]
    final_response: Optional[dict]


def node_analyze_query(state: QAState) -> dict:
    """
    Node 1 (Phase 2): Uses recent chat_history to resolve follow-up references
    (e.g., 'explain that part again') and routes between 'summary' vs 'qa'.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    analyzer = llm.with_structured_output(QueryAnalysis)

    # Format the last 6 messages of chat history so follow-ups have context
    recent_history = state.get("chat_history", [])[-6:]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent_history
    ) or "No prior conversation."

    prompt = (
        "Analyze the user's latest message in the context of the conversation history.\n"
        "1. Determine if the user wants a full/broad 'summary' of the video, or specific 'qa' (including follow-up questions).\n"
        "2. Rewrite the user's message into a clear, standalone search query by resolving any pronouns ('it', 'that', 'the second topic') from chat history."
    )

    analysis: QueryAnalysis = analyzer.invoke(
        [
            ("system", prompt),
            ("human", f"Chat History:\n{history_text}\n\nLatest User Message: {state['question']}"),
        ]
    )

    return {
        "intent": analysis.intent,
        "standalone_query": analysis.standalone_query,
    }


def node_prepare_summary_context(state: QAState) -> dict:
    """Branch A (Summary): Loads all chronological transcript chunks from the video."""
    return {"context_docs": state["documents"]}


def node_retrieve_qa_context(state: QAState) -> dict:
    """Branch B (Q&A / Follow-up): Searches FAISS using the standalone context-resolved query."""
    vectorstore: FAISS = state["vectorstore"]
    docs = vectorstore.similarity_search(state["standalone_query"], k=5)
    return {"context_docs": docs}


def route_intent(state: QAState) -> str:
    """Conditional Router: Routes full summaries to all chunks, and specific Q&A to FAISS."""
    return "summary" if state.get("intent") == "summary" else "qa"


def node_generate_grounded_response(state: QAState) -> dict:
    """
    Convergence Node (Phase 2): Generates a comprehensive English answer or summary
    using ONLY the video transcript excerpts + chat history. Strictly refuses outside info.
    """
    docs = state["context_docs"]
    video_id = state["video_id"]
    orig_lang = state["original_language"]
    intent = state["intent"]

    # Format chunks with exact timestamp metadata
    context_blocks = [
        f"- Timestamp: {d.metadata['timestamp']} (start_sec={d.metadata['start_sec']}) | Content: {d.page_content}"
        for d in docs
    ]
    formatted_context = "\n".join(context_blocks)

    # Include recent chat history so conversational follow-ups flow naturally
    recent_history = state.get("chat_history", [])[-6:]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent_history
    ) or "None"

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)
    structured_llm = llm.with_structured_output(AssistantResponse)

    system_prompt = (
        "You are a helpful, strictly grounded YouTube video assistant. Always respond in clear English.\n"
        "CRITICAL RULES:\n"
        "1. Base your response EXCLUSIVELY on the provided Video Transcript Excerpts. Do NOT use outside knowledge.\n"
        "2. If the user asks about a topic, concept, or detail that is NOT mentioned or taught in the video excerpts, "
        "set 'found_in_video' to False, leave 'citations' empty, and respond clearly: "
        "'This topic is not found in this video.'\n"
        "3. If 'intent' is 'summary', provide a well-structured overview covering the main sections and key takeaways "
        "of the video in chronological order, along with key milestone timestamps in 'citations'.\n"
        "4. If 'intent' is 'qa', answer the user's question thoroughly (referencing prior chat context if it's a follow-up) "
        "and cite the exact 'timestamp' and 'start_sec' where those points were taught."
    )

    user_prompt = (
        f"Intent: {intent}\n"
        f"Original Video Language: {orig_lang}\n"
        f"Recent Conversation History:\n{history_text}\n\n"
        f"Current User Request: {state['question']}\n\n"
        f"Video Transcript Excerpts:\n{formatted_context}"
    )

    result: AssistantResponse = structured_llm.invoke(
        [("system", system_prompt), ("human", user_prompt)]
    )

    # Attach clickable YouTube links (&t=Xs) to every citation
    enriched_citations = [
        {
            "timestamp": c.timestamp,
            "start_sec": c.start_sec,
            "topic_note": c.topic_note,
            "url": f"https://www.youtube.com/watch?v={video_id}&t={c.start_sec}s",
        }
        for c in result.citations
    ]

    return {
        "final_response": {
            "found_in_video": result.found_in_video,
            "response_type": result.response_type,
            "original_language": orig_lang,
            "answer": result.answer,
            "citations": enriched_citations,
        }
    }


def build_qa_graph():
    """Compiles the Phase 2 Conversational Q&A & Summarization LangGraph workflow."""
    builder = StateGraph(QAState)

    builder.add_node("analyze_query", node_analyze_query)
    builder.add_node("prepare_summary_context", node_prepare_summary_context)
    builder.add_node("retrieve_qa_context", node_retrieve_qa_context)
    builder.add_node("generate_grounded_response", node_generate_grounded_response)

    builder.add_edge(START, "analyze_query")
    builder.add_conditional_edges(
        "analyze_query",
        route_intent,
        {
            "summary": "prepare_summary_context",
            "qa": "retrieve_qa_context",
        },
    )
    builder.add_edge("prepare_summary_context", "generate_grounded_response")
    builder.add_edge("retrieve_qa_context", "generate_grounded_response")
    builder.add_edge("generate_grounded_response", END)

    return builder.compile()


qa_app = build_qa_graph()


# ==============================================================================
# 5. LOCAL CLI TEST (Test Summaries, Follow-ups, and Out-of-Scope Topics)
# ==============================================================================
if __name__ == "__main__":
    url = input("Enter a YouTube URL: ").strip()

    # Step A: Run Phase 1 Ingestion
    ingested = ingestion_app.invoke(
        {"video_url": url},
        config={"run_name": "Phase1_Multilingual_Ingestion"},
    )

    print(f"\nVideo Ready! Original Transcript Language: {ingested['original_language']} (Source: {ingested['source']})")
    chat_history = []

    # Step B: Multi-turn conversation loop
    while True:
        q = input("\nYou (ask for a summary, a topic, a follow-up, or 'exit'): ").strip()
        if q.lower() == "exit":
            break

        qa_out = qa_app.invoke(
            {
                "question": q,
                "chat_history": chat_history,
                "video_id": ingested["video_id"],
                "original_language": ingested["original_language"],
                "documents": ingested["documents"],
                "vectorstore": ingested["vectorstore"],
            },
            config={"run_name": "Phase2_Conversational_Video_Assistant"},
        )

        ans = qa_out["final_response"]
        print("\n--- Assistant Response ---")
        if ans["original_language"].lower() != "english":
            print(f"[Language Note]: Originally in {ans['original_language']} (Translated to English)")
        print(f"Found in Video? : {'YES' if ans['found_in_video'] else 'NO'}")
        print(f"Response        :\n{ans['answer']}")
        for cit in ans["citations"]:
            print(f" -> [{cit['timestamp']}] {cit['topic_note']} ({cit['url']})")

        # Save turn to chat_history so the next question can reference this conversation
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": ans["answer"]})