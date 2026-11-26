from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Import the standardized extractors we built in Step 1
from step1_extractors import (
    extract_video_id,
    fetch_youtube_segments,
    transcribe_audio_segments,
)

load_dotenv()

# 1. TIMESTAMP-AWARE CHUNKER
def create_timestamped_chunks(
    segments: list[dict],
    video_id: str,
    max_chars: int = 600,
) -> list[Document]:
    """
    Groups consecutive segments into ~600-character chunks while preserving
    the exact starting timestamp of each chunk in Document.metadata.
    """
    documents = []
    current_texts = []
    current_length = 0
    chunk_start_sec = None
    chunk_start_ts = None

    for i, seg in enumerate(segments):
        if not current_texts:
            chunk_start_sec = seg["start_sec"]
            chunk_start_ts = seg["timestamp"]

        current_texts.append(seg["text"])
        current_length += len(seg["text"]) + 1

        # Once the chunk reaches max_chars (or we hit the final segment), save it
        if current_length >= max_chars or i == len(segments) - 1:
            combined_text = " ".join(current_texts)
            doc = Document(
                page_content=f"[{chunk_start_ts}] {combined_text}",
                metadata={
                    "video_id": video_id,
                    "start_sec": chunk_start_sec,
                    "timestamp": chunk_start_ts,
                    "youtube_link": f"https://www.youtube.com/watch?v={video_id}&t={chunk_start_sec}s",
                },
            )
            documents.append(doc)

            # Keep the last segment as overlap so concepts split across boundaries stay searchable
            if i < len(segments) - 1:
                current_texts = [seg["text"]]
                current_length = len(seg["text"])
                chunk_start_sec = seg["start_sec"]
                chunk_start_ts = seg["timestamp"]
            else:
                current_texts = []

    return documents

# 2. EMBED & BUILD FAISS INDEX
def build_faiss_index(documents: list[Document]) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# --- TEST STEP 2 ---
if __name__ == "__main__":
    url = input("Enter a YouTube URL: ").strip()
    vid_id = extract_video_id(url)

    print("\n1. Extracting timestamped segments...")
    segments = fetch_youtube_segments(url)
    if not segments:
        print("Captions unavailable -> running Gemini audio fallback...")
        segments = transcribe_audio_segments(url)

    print(f"2. Chunking {len(segments)} segments while preserving timestamps...")
    docs = create_timestamped_chunks(segments, video_id=vid_id, max_chars=600)
    print(f"   Created {len(docs)} chunks. Sample chunk metadata: {docs[0].metadata}")

    print("3. Embedding chunks into FAISS...")
    vectorstore = build_faiss_index(docs)

    # Test similarity retrieval
    test_query = input("\nEnter a topic/question to search in the video: ").strip()
    results = vectorstore.similarity_search_with_score(test_query, k=3)

    print("\n--- Top 3 Matching Chunks ---")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\n#{rank} | Score (L2 distance): {score:.4f} | Time: {doc.metadata['timestamp']}")
        print(f"Link: {doc.metadata['youtube_link']}")
        print(f"Text: {doc.page_content[:200]}...")