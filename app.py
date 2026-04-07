from __future__ import annotations

import base64
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chat import (  # noqa: E402
    DEFAULT_MODE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_STORAGE_DIR,
    answer_question,
    format_page_range,
    load_index,
)
from conversation_store import ConversationStore  # noqa: E402
from export_utils import (  # noqa: E402
    conversation_markdown,
    conversation_pdf_bytes,
    slugify_filename,
)

COURSE_OPTIONS = [
    "All",
    "Adjustment Theory I",
    "Geodatabases",
    "Photogrametric CV",
    "Intr. Space Geodesy",
    "Geoinformatics",
]
STUDY_TASK_OPTIONS = {
    "Ask a question": "ask",
    "Summarize lecture": "summarize-lecture",
    "Generate exam questions": "generate-exam-questions",
}
MODE_OPTIONS = {
    "Professor explanation": "hybrid-professor",
    "Strict local only": "strict",
}
CHAT_DB_PATH = DEFAULT_STORAGE_DIR / "chat_history.sqlite3"


st.set_page_config(
    page_title="GeoAI Assistant",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-a: #efefec;
            --bg-b: #deddd8;
            --bg-c: #cdccc6;
            --panel: rgba(248, 248, 245, 0.78);
            --panel-solid: #fbfbf8;
            --panel-soft: #ecece8;
            --line: rgba(25, 26, 26, 0.10);
            --line-strong: rgba(25, 26, 26, 0.20);
            --ink: #171818;
            --muted: #666863;
            --muted-soft: #878983;
            --accent: #232423;
            --accent-soft: #d6d5d0;
            --user: linear-gradient(145deg, #313232, #171818);
            --assistant: linear-gradient(180deg, #fbfbf8, #eeeeea);
            --shadow-soft: 0 24px 54px rgba(20, 21, 21, 0.08);
            --shadow-deep: 0 34px 70px rgba(20, 21, 21, 0.12);
            --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", "Baskerville", Georgia, serif;
            --sans: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }

        html, body, [class*="css"] {
            font-family: var(--sans);
        }

        ::selection {
            background: rgba(23, 24, 24, 0.12);
            color: inherit;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.68), transparent 22%),
                radial-gradient(circle at 82% 0%, rgba(68, 69, 69, 0.08), transparent 26%),
                linear-gradient(180deg, var(--bg-a), var(--bg-b) 56%, var(--bg-c));
            color: var(--ink);
        }

        .block-container {
            max-width: 1440px;
            padding-top: 0.9rem;
            padding-bottom: 2.2rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(244, 244, 241, 0.96), rgba(228, 228, 224, 0.98));
            border-right: 1px solid var(--line);
            box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.44);
        }

        input[type="checkbox"],
        input[type="radio"] {
            accent-color: var(--accent) !important;
        }

        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stToggle"] label,
        [data-baseweb="radio"],
        [data-baseweb="checkbox"] {
            background: transparent !important;
            box-shadow: none !important;
        }

        [data-testid="stRadio"] label > div,
        [data-testid="stCheckbox"] label > div,
        [data-testid="stToggle"] label > div {
            background: transparent !important;
        }

        [data-testid="stRadio"] p,
        [data-testid="stCheckbox"] p,
        [data-testid="stToggle"] p {
            background: transparent !important;
            box-shadow: none !important;
        }

        label[data-testid="stWidgetLabel"] p {
            color: var(--muted) !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.16em;
        }

        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: inherit;
        }

        .sidebar-brand {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at top right, rgba(255,255,255,0.08), transparent 28%),
                linear-gradient(160deg, rgba(32, 33, 33, 0.98), rgba(17, 18, 18, 0.98));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 28px 28px 16px 16px;
            padding: 1.15rem 1.05rem 1.05rem 1.05rem;
            margin-bottom: 0.95rem;
            box-shadow: var(--shadow-deep);
        }

        .sidebar-brand::before {
            content: "";
            position: absolute;
            top: 15px;
            left: 16px;
            width: 92px;
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0.52), transparent);
        }

        .sidebar-kicker {
            color: rgba(255, 255, 255, 0.58);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            margin-bottom: 0.52rem;
            padding-top: 0.25rem;
        }

        .sidebar-title {
            color: #f6f6f3;
            font-family: var(--serif);
            font-size: 1.62rem;
            font-weight: 700;
            line-height: 1.04;
            letter-spacing: -0.03em;
            margin-bottom: 0.4rem;
        }

        .sidebar-copy {
            color: rgba(255, 255, 255, 0.72);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at top right, rgba(255,255,255,0.08), transparent 26%),
                linear-gradient(160deg, rgba(19, 20, 20, 0.98), rgba(40, 41, 41, 0.96) 56%, rgba(17, 18, 18, 0.99));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 34px 34px 18px 18px;
            padding: 1.55rem 1.55rem 1.35rem 1.55rem;
            box-shadow: var(--shadow-deep);
            backdrop-filter: blur(12px);
            margin-bottom: 1rem;
        }

        .hero-card::before {
            content: "";
            position: absolute;
            top: 18px;
            left: 22px;
            width: 96px;
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0.54), transparent);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            right: -28px;
            bottom: -44px;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.10), transparent 63%);
        }

        .hero-kicker {
            color: rgba(255,255,255,0.58);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.22em;
            margin-bottom: 0.6rem;
            padding-top: 0.2rem;
        }

        .hero-title {
            color: #f8f8f5;
            font-family: var(--serif);
            font-size: clamp(2.7rem, 4vw, 4.4rem);
            font-weight: 700;
            line-height: 0.92;
            letter-spacing: -0.045em;
            margin-bottom: 0.7rem;
            max-width: 7ch;
            text-wrap: balance;
        }

        .hero-copy {
            color: rgba(255,255,255,0.72);
            font-size: 1.02rem;
            line-height: 1.72;
            max-width: 760px;
        }

        .hero-divider {
            width: 100%;
            height: 1px;
            margin: 1.05rem 0 0.95rem 0;
            background: linear-gradient(90deg, rgba(255,255,255,0.18), transparent 76%);
        }

        .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.52rem;
        }

        .status-pill {
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.12);
            color: rgba(255,255,255,0.88);
            padding: 0.42rem 0.74rem;
            border-radius: 999px;
            font-size: 0.84rem;
            backdrop-filter: blur(10px);
        }

        .sidebar-section-title {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            color: var(--muted-soft);
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }

        .conversation-meta {
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: -0.06rem;
            margin-bottom: 0.62rem;
            line-height: 1.35;
        }

        div[data-testid="stChatMessage"] {
            background: var(--assistant);
            border: 1px solid rgba(25, 26, 26, 0.08);
            border-radius: 30px 30px 30px 10px;
            box-shadow: 0 20px 38px rgba(20, 21, 21, 0.07);
            padding: 0.42rem 0.56rem;
            margin-bottom: 0.92rem;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: var(--user);
            border-color: rgba(255, 255, 255, 0.08);
            border-radius: 30px 30px 10px 30px;
            box-shadow: 0 24px 44px rgba(7, 8, 8, 0.18);
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            padding: 0.1rem 0.15rem;
        }

        [data-testid="stChatMessageAvatarAssistant"],
        [data-testid="stChatMessageAvatarUser"] {
            background: rgba(250, 250, 247, 0.92) !important;
            color: var(--accent) !important;
            border: 1px solid rgba(25, 26, 26, 0.10) !important;
            box-shadow: 0 10px 20px rgba(20, 21, 21, 0.08);
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
            background: rgba(31, 32, 32, 0.98) !important;
            color: #f8f8f5 !important;
            border-color: rgba(255, 255, 255, 0.12) !important;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li {
            line-height: 1.68;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) li,
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) strong {
            color: white !important;
        }

        .section-card {
            background: linear-gradient(180deg, rgba(251, 251, 248, 0.96), rgba(238, 238, 234, 0.92));
            border: 1px solid rgba(25, 26, 26, 0.08);
            border-radius: 28px 28px 16px 16px;
            padding: 1.05rem;
            box-shadow: var(--shadow-soft);
        }

        .section-title {
            font-family: var(--serif);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.22rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .source-title {
            font-size: 0.84rem;
            color: var(--muted);
            margin-top: 0.65rem;
            margin-bottom: 0.2rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .source-caption {
            color: var(--muted);
            font-size: 0.79rem;
            margin-top: 0.2rem;
            margin-bottom: 0.55rem;
        }

        .empty-state {
            color: var(--muted);
            line-height: 1.6;
            padding: 1.3rem 0.1rem 0.35rem 0.1rem;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stLinkButton > a {
            border-radius: 14px !important;
            border: 1px solid var(--line-strong) !important;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(237, 237, 232, 0.92)) !important;
            color: var(--ink) !important;
            font-weight: 600 !important;
            min-height: 2.45rem !important;
            box-shadow: 0 14px 24px rgba(20, 21, 21, 0.05) !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stLinkButton > a:hover {
            border-color: rgba(35, 36, 35, 0.30) !important;
            color: var(--accent) !important;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(242, 242, 239, 0.96)) !important;
            transform: translateY(-1px);
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        .stTextInput > div > div,
        .stTextArea textarea {
            background: rgba(250, 250, 247, 0.88) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.62);
        }

        [data-baseweb="select"] > div:focus-within,
        [data-baseweb="input"] > div:focus-within,
        .stTextInput > div > div:focus-within,
        .stTextArea textarea:focus {
            border-color: var(--line-strong) !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.72),
                0 0 0 1px rgba(35, 36, 35, 0.08) !important;
        }

        [data-testid="stExpander"] {
            background: rgba(249, 249, 246, 0.90);
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
        }

        [data-testid="stExpander"] summary {
            color: var(--ink) !important;
            font-weight: 600;
        }

        [data-testid="stChatInput"] {
            background: linear-gradient(180deg, rgba(251, 251, 248, 0.96), rgba(239, 239, 235, 0.94));
            border: 1px solid rgba(25, 26, 26, 0.12);
            border-radius: 28px;
            box-shadow: var(--shadow-soft);
        }

        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: var(--ink) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_index(storage_dir: str):
    return load_index(Path(storage_dir).resolve())


@st.cache_resource(show_spinner=False)
def get_store(db_path: str) -> ConversationStore:
    return ConversationStore(Path(db_path).resolve())


@st.cache_data(show_spinner=False)
def read_binary_file(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


@st.cache_data(show_spinner=False)
def read_text_preview(path_str: str, limit: int = 12000) -> str:
    path = Path(path_str)
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)[:limit]
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")[:limit]


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = None
    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None


def format_user_prompt(study_task: str, text: str) -> str:
    if study_task == "summarize-lecture":
        return f"Summarize lecture: {text}"
    if study_task == "generate-exam-questions":
        return f"Generate exam questions: {text}"
    return text


def task_placeholder(study_task: str) -> str:
    if study_task == "summarize-lecture":
        return "Type a lecture number, file topic, or theme to summarize..."
    if study_task == "generate-exam-questions":
        return "Type a topic or lecture for exam question generation..."
    return "Ask about a concept, theorem, lecture, or formula..."


def build_conversation_title(text: str) -> str:
    clean = " ".join(text.strip().split())
    prefixes = ("Summarize lecture:", "Generate exam questions:")
    for prefix in prefixes:
        if clean.startswith(prefix):
            clean = clean[len(prefix) :].strip()
            break
    if not clean:
        return "New chat"
    return clean[:58] + ("..." if len(clean) > 58 else "")


def format_timestamp(raw_value: str) -> str:
    try:
        moment = datetime.fromisoformat(raw_value)
    except ValueError:
        return raw_value
    return moment.astimezone().strftime("%d %b | %H:%M")


def current_conversation_record(
    conversations: list[dict[str, object]],
    active_conversation_id: int | None,
) -> dict[str, object] | None:
    if active_conversation_id is None:
        return None
    for conversation in conversations:
        if int(conversation["id"]) == active_conversation_id:
            return conversation
    return None


def load_conversation_into_session(store: ConversationStore, conversation_id: int) -> None:
    st.session_state.active_conversation_id = conversation_id
    st.session_state.messages = store.get_messages(conversation_id)
    st.session_state.selected_source = None


def ensure_active_conversation(store: ConversationStore) -> None:
    if st.session_state.active_conversation_id is not None:
        return

    conversations = store.list_conversations(limit=1)
    if conversations:
        load_conversation_into_session(store, int(conversations[0]["id"]))
        return

    conversation_id = store.create_conversation()
    load_conversation_into_session(store, conversation_id)


def start_new_conversation(store: ConversationStore) -> None:
    conversation_id = store.create_conversation()
    load_conversation_into_session(store, conversation_id)


def delete_active_conversation(store: ConversationStore) -> None:
    active_id = st.session_state.active_conversation_id
    if active_id is None:
        return

    store.delete_conversation(active_id)
    remaining = store.list_conversations(limit=1)
    if remaining:
        load_conversation_into_session(store, int(remaining[0]["id"]))
    else:
        start_new_conversation(store)


def collect_sources(results: list[dict[str, object]]) -> list[dict[str, object]]:
    sources_by_path: dict[str, dict[str, object]] = {}

    for item in results:
        source_path = str(item["source_path"])
        source = sources_by_path.setdefault(
            source_path,
            {
                "source_path": source_path,
                "source_relpath": str(item["source_relpath"]),
                "course_name": str(item["course_name"]),
                "page_labels": [],
            },
        )
        page_label = format_page_range(item.get("page_start"), item.get("page_end"))
        if page_label and page_label not in source["page_labels"]:
            source["page_labels"].append(page_label)

    return list(sources_by_path.values())


def render_source_buttons(message_index: int, sources: list[dict[str, object]]) -> None:
    if not sources:
        return

    st.markdown('<div class="source-title">Sources</div>', unsafe_allow_html=True)
    for row_start in range(0, len(sources), 2):
        row_sources = sources[row_start : row_start + 2]
        columns = st.columns(2, gap="small")
        for column, source in zip(columns, row_sources):
            file_label = Path(str(source["source_relpath"])).name
            meta = [str(source["course_name"])]
            if source["page_labels"]:
                meta.append(", ".join(source["page_labels"]))

            with column:
                if st.button(
                    file_label,
                    key=f"source-{message_index}-{row_start}-{file_label}",
                    use_container_width=True,
                ):
                    st.session_state.selected_source = source
                st.markdown(
                    f'<div class="source-caption">{" | ".join(meta)}</div>',
                    unsafe_allow_html=True,
                )


def render_source_preview() -> None:
    selected_source = st.session_state.selected_source
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Source preview</div>
            <div class="section-copy">Open the original lecture file directly inside the app.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not selected_source:
        st.markdown(
            '<div class="empty-state">Select any source under an answer to preview the original PDF or text file here.</div>',
            unsafe_allow_html=True,
        )
        return

    source_path = Path(str(selected_source["source_path"]))
    st.caption(str(selected_source["source_relpath"]))
    if selected_source["page_labels"]:
        st.markdown(
            f'<div class="conversation-meta">{", ".join(selected_source["page_labels"])}</div>',
            unsafe_allow_html=True,
        )

    if not source_path.exists():
        st.warning("The selected source file could not be found on disk.")
        return

    if source_path.suffix.lower() == ".pdf":
        pdf_base64 = base64.b64encode(read_binary_file(str(source_path))).decode("ascii")
        st.markdown(
            (
                f'<iframe src="data:application/pdf;base64,{pdf_base64}" '
                'width="100%" height="820" '
                'style="border:none;border-radius:16px;background:white;"></iframe>'
            ),
            unsafe_allow_html=True,
        )
    else:
        st.code(read_text_preview(str(source_path)))

    st.link_button("Open file externally", source_path.resolve().as_uri(), use_container_width=True)


def render_message(message_index: int, message: dict[str, object]) -> None:
    role = str(message["role"])

    with st.chat_message(role):
        st.markdown(str(message["content"]))
        if role == "assistant":
            render_source_buttons(message_index, list(message.get("sources", [])))


def render_conversation_list(
    store: ConversationStore,
    conversations: list[dict[str, object]],
) -> None:
    if not conversations:
        st.markdown(
            '<div class="empty-state">No conversations match your search.</div>',
            unsafe_allow_html=True,
        )
        return

    for conversation in conversations:
        conversation_id = int(conversation["id"])
        title = str(conversation["title"])
        is_active = conversation_id == st.session_state.active_conversation_id
        button_label = f"{'Current | ' if is_active else ''}{title}"
        if st.button(
            button_label,
            key=f"conversation-{conversation_id}",
            use_container_width=True,
        ):
            load_conversation_into_session(store, conversation_id)
            st.rerun()

        meta = f"{format_timestamp(str(conversation['updated_at']))} | {conversation['message_count']} messages"
        st.markdown(f'<div class="conversation-meta">{meta}</div>', unsafe_allow_html=True)


def render_current_chat_controls(
    store: ConversationStore,
    current_record: dict[str, object] | None,
) -> None:
    if not current_record:
        return

    conversation_id = int(current_record["id"])
    current_title = str(current_record["title"])
    title_key = f"conversation-title-{conversation_id}"

    st.markdown('<div class="sidebar-section-title">Current chat</div>', unsafe_allow_html=True)
    st.text_input("Conversation title", value=current_title, key=title_key)

    save_col, delete_col = st.columns(2, gap="small")
    with save_col:
        if st.button("Rename", key=f"rename-{conversation_id}", use_container_width=True):
            store.rename_conversation(conversation_id, st.session_state[title_key])
            st.rerun()

    with delete_col:
        if st.button("Delete", key=f"delete-{conversation_id}", use_container_width=True):
            delete_active_conversation(store)
            st.rerun()

    export_base = slugify_filename(current_title)
    markdown_payload = conversation_markdown(current_title, st.session_state.messages)
    pdf_payload = conversation_pdf_bytes(current_title, st.session_state.messages)

    st.download_button(
        "Export Markdown",
        data=markdown_payload,
        file_name=f"{export_base}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.download_button(
        "Export Styled PDF",
        data=pdf_payload,
        file_name=f"{export_base}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def render_sidebar(store: ConversationStore) -> tuple[dict[str, object], list[dict[str, object]]]:
    all_conversations = store.list_conversations(limit=100)

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-kicker">Local RAG + Ollama</div>
                <div class="sidebar-title">GeoAI Assistant</div>
                <div class="sidebar-copy">
                    A fully local study companion for Geodesy and Geoinformation Science, with saved conversations,
                    grounded retrieval, and a cleaner professor-style workflow.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("New chat", use_container_width=True):
            start_new_conversation(store)
            st.rerun()

        st.markdown('<div class="sidebar-section-title">Search history</div>', unsafe_allow_html=True)
        search_query = st.text_input(
            "Search conversations",
            placeholder="Search title or message text...",
            label_visibility="collapsed",
        )

        visible_conversations = store.list_conversations(limit=100, search_query=search_query)

        st.markdown('<div class="sidebar-section-title">Conversations</div>', unsafe_allow_html=True)
        render_conversation_list(store, visible_conversations)
        render_current_chat_controls(
            store,
            current_conversation_record(all_conversations, st.session_state.active_conversation_id),
        )

        st.markdown('<div class="sidebar-section-title">Study settings</div>', unsafe_allow_html=True)
        course = st.selectbox("Course filter", COURSE_OPTIONS)
        study_task_label = st.radio("Study mode", list(STUDY_TASK_OPTIONS), index=0)
        default_mode_index = list(MODE_OPTIONS.values()).index(DEFAULT_MODE)
        mode_label = st.radio("Grounding mode", list(MODE_OPTIONS), index=default_mode_index)
        beginner_mode = st.toggle("Beginner mode", value=False)

        with st.expander("Advanced", expanded=False):
            top_k = st.slider("Top retrieved passages", min_value=3, max_value=8, value=5)
            min_score = st.slider(
                "Minimum retrieval score",
                min_value=0.0,
                max_value=0.08,
                value=0.015,
                step=0.005,
            )

        st.caption(f"Stored locally in `{CHAT_DB_PATH.name}`")

    settings = {
        "course": course,
        "study_task": STUDY_TASK_OPTIONS[study_task_label],
        "mode": MODE_OPTIONS[mode_label],
        "beginner_mode": beginner_mode,
        "top_k": top_k,
        "min_score": min_score,
    }
    return settings, all_conversations


def render_header(settings: dict[str, object], conversations: list[dict[str, object]]) -> None:
    current_record = current_conversation_record(conversations, st.session_state.active_conversation_id)
    title = str(current_record["title"]) if current_record else "New chat"
    course_label = "All courses" if settings["course"] == "All" else str(settings["course"])
    task_label = next(label for label, value in STUDY_TASK_OPTIONS.items() if value == settings["study_task"])
    mode_label = next(label for label, value in MODE_OPTIONS.items() if value == settings["mode"])
    beginner_label = "Beginner mode on" if settings["beginner_mode"] else "Master's level"

    st.markdown(
        (
            '<div class="hero-card">'
            '<div class="hero-kicker">GeoAI Assistant | Local RAG + Ollama</div>'
            '<div class="hero-title">GeoAI Assistant</div>'
            '<div class="hero-copy">'
            "GeoAI Assistant is a fully local study studio for a Master's in Geodesy and Geoinformation Science. "
            "It grounds answers in your own materials, keeps the full conversation history on your machine, and supports professor-style explanations, lecture synthesis, and exam preparation."
            "</div>"
            '<div class="hero-divider"></div>'
            '<div class="status-strip">'
            f'<div class="status-pill">Chat: {title}</div>'
            f'<div class="status-pill">{task_label}</div>'
            f'<div class="status-pill">{mode_label}</div>'
            f'<div class="status-pill">{course_label}</div>'
            f'<div class="status-pill">{beginner_label}</div>'
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_custom_styles()
    init_session_state()

    store = get_store(str(CHAT_DB_PATH))
    ensure_active_conversation(store)
    settings, conversations = render_sidebar(store)
    render_header(settings, conversations)

    try:
        index = get_index(str(DEFAULT_STORAGE_DIR))
    except Exception as exc:
        st.error(f"Could not load the local index from {DEFAULT_STORAGE_DIR}: {exc}")
        st.info("Run `python3 src/ingest.py` first if the index has not been created yet.")
        return

    main_col, preview_col = st.columns([1.72, 1.02], gap="large")

    with main_col:
        if not st.session_state.messages:
            st.markdown(
                '<div class="empty-state">Start a conversation and it will be saved automatically. '
                "Follow-up questions in the same chat can reuse recent context.</div>",
                unsafe_allow_html=True,
            )

        for message_index, message in enumerate(st.session_state.messages):
            render_message(message_index, message)

        prompt = st.chat_input(task_placeholder(str(settings["study_task"])))

    if prompt:
        question = prompt.strip()
        prior_history = list(st.session_state.messages)
        display_question = format_user_prompt(str(settings["study_task"]), question)
        st.session_state.selected_source = None

        store.add_message(
            st.session_state.active_conversation_id,
            "user",
            display_question,
            [],
        )
        if not prior_history:
            store.rename_conversation(
                st.session_state.active_conversation_id,
                build_conversation_title(display_question),
            )

        st.session_state.messages.append({"role": "user", "content": display_question, "sources": []})

        with st.spinner("Thinking through your materials..."):
            try:
                answer, results = answer_question(
                    question=question,
                    index=index,
                    model=DEFAULT_MODEL,
                    ollama_host=DEFAULT_OLLAMA_HOST,
                    mode=str(settings["mode"]),
                    study_task=str(settings["study_task"]),
                    beginner_mode=bool(settings["beginner_mode"]),
                    top_k=int(settings["top_k"]),
                    min_score=float(settings["min_score"]),
                    course_filters=[] if settings["course"] == "All" else [str(settings["course"])],
                    folder_filters=[],
                    show_retrieved=False,
                    conversation_history=prior_history,
                )
                sources = collect_sources(results)
                assistant_message = {"role": "assistant", "content": answer, "sources": sources}
                st.session_state.messages.append(assistant_message)
                store.add_message(
                    st.session_state.active_conversation_id,
                    "assistant",
                    answer,
                    sources,
                )
            except Exception as exc:
                error_message = f"Error: {exc}"
                assistant_message = {"role": "assistant", "content": error_message, "sources": []}
                st.session_state.messages.append(assistant_message)
                store.add_message(
                    st.session_state.active_conversation_id,
                    "assistant",
                    error_message,
                    [],
                )
        st.rerun()

    with preview_col:
        render_source_preview()


if __name__ == "__main__":
    main()
