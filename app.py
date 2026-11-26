import streamlit as st
from final_backend import ingestion_app, qa_app

# ==============================================================================
# 1. PAGE CONFIGURATION & SESSION STATE INITIALIZATION
# ==============================================================================

st.set_page_config(
    page_title="YouTube Timestamp Q&A & Verification",
    page_icon="🎥",
    layout="wide",
)

# Initialize persistent variables in Streamlit's session_state so data survives
# across button clicks and multi-turn chat interactions.
if "ingested_data" not in st.session_state:
    st.session_state.ingested_data = None      # Stores Phase 1 output (vectorstore, documents, video_id, etc.)
if "messages" not in st.session_state:
    st.session_state.messages = []             # Stores full chat history for display & Phase 2 context
if "video_start_sec" not in st.session_state:
    st.session_state.video_start_sec = 0       # Controls where the embedded YouTube player starts playing


# ==============================================================================
# 2. SIDEBAR: YOUTUBE URL INPUT & INGESTION CONTROLS
# ==============================================================================

with st.sidebar:
    st.header("1. Video Ingestion")
    video_url = st.text_input(
        "Paste YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    process_btn = st.button("Process Video", type="primary", use_container_width=True)

    if process_btn:
        if not video_url.strip():
            st.warning("Please enter a valid YouTube URL first.")
        else:
            with st.spinner("Extracting transcript, translating (if needed), and building FAISS index..."):
                try:
                    # Run Phase 1 LangGraph Ingestion Workflow once for this URL
                    result = ingestion_app.invoke(
                        {"video_url": video_url.strip()},
                        config={"run_name": "Streamlit_Phase1_Ingestion"},
                    )
                    # Save the compiled FAISS index and metadata into session_state
                    st.session_state.ingested_data = result
                    st.session_state.messages = []       # Reset chat history for the new video
                    st.session_state.video_start_sec = 0 # Reset video player to 00:00
                    st.success("Video processed and indexed!")
                except Exception as e:
                    st.error(f"Failed to process video: {e}")

    # Display metadata summary once a video has been ingested
    if st.session_state.ingested_data:
        data = st.session_state.ingested_data
        st.divider()
        st.subheader("Video Metadata")
        st.markdown(f"**Video ID:** `{data['video_id']}`")
        st.markdown(f"**Original Language:** `{data['original_language']}`")

        source_label = (
            "YouTube Captions API"
            if data["source"] == "youtube_api"
            else "Audio Fallback (yt-dlp + Gemini)"
        )
        st.markdown(f"**Extraction Path:** `{source_label}`")
        st.markdown(f"**Indexed Chunks:** `{len(data['documents'])}`")

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ==============================================================================
# 3. MAIN WORKSPACE: VIDEO PLAYER (LEFT) & CONVERSATIONAL Q&A (RIGHT)
# ==============================================================================

st.title("YouTube Timestamp Q&A & Topic Verification")
st.caption("Ask for a full video summary, verify if a topic was taught, and jump directly to exact timestamps.")

if not st.session_state.ingested_data:
    st.info("Paste a YouTube URL in the left sidebar and click **Process Video** to begin.")
    st.stop()

data = st.session_state.ingested_data

# Show a clear notice if the original video was in another language (e.g., Hindi)
if data["original_language"].lower() != "english":
    st.info(
        f"**Language Notice:** This video was originally in **{data['original_language']}**. "
        "It has been automatically translated and indexed in **English**."
    )

# Split the main screen into two columns: Video Player on left, Chat Interface on right
col_video, col_chat = st.columns([1, 1.3], gap="large")

with col_video:
    st.subheader("Video Preview")
    # Embed the YouTube player starting at st.session_state.video_start_sec
    st.video(
        f"https://www.youtube.com/watch?v={data['video_id']}",
        start_time=st.session_state.video_start_sec,
    )
    st.caption(f"Currently cued at: **{st.session_state.video_start_sec}s** (Click any timestamp button on the right to jump)")

with col_chat:
    st.subheader("Ask About This Video")

    # Quick-action prompt buttons for convenience
    q_col1, q_col2 = st.columns(2)
    quick_prompt = None
    if q_col1.button("Summarize this video", use_container_width=True):
        quick_prompt = "Summarize this video with key timestamps."
    if q_col2.button("List main topics taught", use_container_width=True):
        quick_prompt = "What are the main topics taught in this video and when do they start?"

    # Render previous conversation turns from session_state
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # If this assistant message includes verified timestamps, render jump buttons
            if msg["role"] == "assistant" and msg.get("citations"):
                st.markdown("**Verified Timestamps:**")
                btn_cols = st.columns(min(len(msg["citations"]), 3))
                for c_idx, cit in enumerate(msg["citations"]):
                    col = btn_cols[c_idx % len(btn_cols)]
                    label = f"▶ [{cit['timestamp']}] {cit['topic_note'][:28]}..."
                    if col.button(label, key=f"jump_{msg_idx}_{c_idx}", help=f"{cit['topic_note']} ({cit['url']})"):
                        st.session_state.video_start_sec = cit["start_sec"]
                        st.rerun()

    # Accept typed user input or quick-action button click
    user_query = st.chat_input("Ask a question, request a summary, or ask a follow-up...")
    active_query = quick_prompt or user_query

    if active_query:
        # 1. Display and store the user's message
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # 2. Build plain-text chat history for Phase 2 query rewriting
        history_for_graph = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        # 3. Invoke Phase 2 LangGraph Q&A & Verification Workflow
        with st.chat_message("assistant"):
            with st.spinner("Searching transcript & verifying timestamps..."):
                qa_out = qa_app.invoke(
                    {
                        "question": active_query,
                        "chat_history": history_for_graph,
                        "video_id": data["video_id"],
                        "original_language": data["original_language"],
                        "documents": data["documents"],
                        "vectorstore": data["vectorstore"],
                    },
                    config={"run_name": "Streamlit_Phase2_QA"},
                )

                ans = qa_out["final_response"]

                # Format response banner based on whether the topic was found in the video
                if not ans["found_in_video"]:
                    formatted_reply = f"**Not Found in Video:** {ans['answer']}"
                else:
                    formatted_reply = ans["answer"]

                st.markdown(formatted_reply)

                # Save assistant turn (including citations) to session_state and rerun so buttons become interactive
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": formatted_reply,
                        "citations": ans["citations"],
                    }
                )
                st.rerun()