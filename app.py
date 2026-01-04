"""
app.py
Streamlit interface for Library Chatbot with FAQ and database recommendations.
Auto-builds indices if not present (for Streamlit Cloud deployment).
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from openai import OpenAI
import re
from typing import List, Tuple
from pathlib import Path
import logging

# Configure
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Library Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
class Config:
    # Source data paths
    FAQ_SOURCE = "library_faq_clean.csv"
    DB_SOURCE = "Databases description.csv"
    
    # Generated index paths
    FAQ_TEXT_PATH = "faq_text.parquet"
    FAQ_EMB_PATH = "faq_emb.npy"
    DB_TEXT_PATH = "db_text.parquet"
    DB_EMB_PATH = "db_emb.npy"
    
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE = 200
    
    FAQ_HIGH_CONFIDENCE = 0.70
    FAQ_MIN_CONFIDENCE = 0.60
    DB_MIN_CONFIDENCE = 0.45
    BOTH_DELTA = 0.06

class IndexBuilder:
    """Builds indices if they don't exist."""
    
    @staticmethod
    def sanitize_matrix(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float32)
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        m = np.clip(m, -10.0, 10.0)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        m = m / np.clip(norms, 1e-12, None)
        return np.ascontiguousarray(m)
    
    @staticmethod
    def embed_batch(client: OpenAI, text_list: List[str], 
                   model: str = Config.EMBEDDING_MODEL,
                   batch_size: int = Config.EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            embeddings.extend([d.embedding for d in resp.data])
        
        arr = np.array(embeddings, dtype=np.float32)
        return IndexBuilder.sanitize_matrix(arr)
    
    @staticmethod
    def indices_exist() -> bool:
        """Check if all index files exist."""
        return all(Path(p).exists() for p in [
            Config.FAQ_TEXT_PATH,
            Config.FAQ_EMB_PATH,
            Config.DB_TEXT_PATH,
            Config.DB_EMB_PATH
        ])
    
    @staticmethod
    def build_indices(client: OpenAI) -> None:
        """Build all indices from source CSV files."""
        # Build FAQ index
        faq = pd.read_csv(Config.FAQ_SOURCE)
        faq["question"] = faq["question"].fillna("").astype(str).str.strip()
        faq["answer"] = faq["answer"].fillna("").astype(str).str.strip()
        faq = faq[(faq["question"] != "") & (faq["answer"] != "")].reset_index(drop=True)
        
        faq_emb = IndexBuilder.embed_batch(client, faq["question"].tolist())
        
        faq[["question", "answer"]].to_parquet(Config.FAQ_TEXT_PATH, index=False)
        np.save(Config.FAQ_EMB_PATH, faq_emb)
        
        # Build Database index
        db = pd.read_csv(Config.DB_SOURCE)
        db["name"] = db["name"].fillna("").astype(str).str.strip()
        db["description"] = db["description"].fillna("").astype(str).str.strip()
        db = db[(db["name"] != "") & (db["description"] != "")].reset_index(drop=True)
        
        db_text = (db["name"] + ". " + db["description"]).tolist()
        db_emb = IndexBuilder.embed_batch(client, db_text)
        
        db[["name", "description"]].to_parquet(Config.DB_TEXT_PATH, index=False)
        np.save(Config.DB_EMB_PATH, db_emb)

class EmbeddingUtils:
    """Utilities for safe embedding operations."""
    
    @staticmethod
    def sanitize_matrix(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float32)
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        m = np.clip(m, -10.0, 10.0)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        m = m / np.clip(norms, 1e-12, None)
        return np.ascontiguousarray(m)
    
    @staticmethod
    def sanitize_vec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.clip(v, -10.0, 10.0)
        n = np.linalg.norm(v)
        v = v / np.clip(n, 1e-12, None)
        return np.ascontiguousarray(v)
    
    @staticmethod
    def clean_html(s: str) -> str:
        s = re.sub(r"<br\s*/?>", "\n", str(s), flags=re.IGNORECASE)
        s = re.sub(r"<[^>]+>", "", s)
        s = re.sub(r"\s+\n", "\n", s)
        return s.strip()

class IntentDetector:
    """Detects user intent from query."""
    
    DB_KEYWORDS = re.compile(
        r"\b(database|db|which database|where to search|where can i find|"
        r"source for|best database|recommend|ieee|scopus|pubmed|jstor|"
        r"proquest|web of science|find.*articles?|search.*papers?|"
        r"research.*source)\b",
        re.IGNORECASE
    )
    
    RESEARCH_TOPICS = re.compile(
        r"\b(articles?|papers?|journals?|conference|proceedings|"
        r"publications?|research|standards?|thesis|dissertation)\b",
        re.IGNORECASE
    )
    
    @classmethod
    def is_database_intent(cls, query: str) -> bool:
        has_db_keywords = bool(cls.DB_KEYWORDS.search(query))
        has_research_topics = bool(cls.RESEARCH_TOPICS.search(query))
        return has_db_keywords or has_research_topics

class Retriever:
    """Handles embedding-based retrieval."""
    
    def __init__(self, embeddings: np.ndarray, df: pd.DataFrame):
        self.embeddings = EmbeddingUtils.sanitize_matrix(embeddings)
        self.df = df
    
    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        query_vec = EmbeddingUtils.sanitize_vec(query_vec)
        
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            similarities = self.embeddings @ query_vec
        
        similarities = np.nan_to_num(similarities, nan=-1.0, posinf=-1.0, neginf=-1.0)
        similarities = np.clip(similarities, -1.0, 1.0)
        
        top_indices = np.argsort(-similarities)[:k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

class LibraryChatbot:
    """Main chatbot class."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
        # Check if indices need to be built
        if not IndexBuilder.indices_exist():
            st.info("🔨 Building indices for first time. This will take 1-2 minutes...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Building FAQ index...")
                progress_bar.progress(25)
                IndexBuilder.build_indices(self.client)
                progress_bar.progress(100)
                status_text.text("✅ Indices built successfully!")
                st.success("Indices created! Starting chatbot...")
            except Exception as e:
                st.error(f"❌ Error building indices: {e}")
                raise
        
        # Load FAQ data
        self.faq_df = pd.read_parquet(Config.FAQ_TEXT_PATH)
        faq_emb = np.load(Config.FAQ_EMB_PATH)
        self.faq_retriever = Retriever(faq_emb, self.faq_df)
        
        # Load database data
        self.db_df = pd.read_parquet(Config.DB_TEXT_PATH)
        db_emb = np.load(Config.DB_EMB_PATH)
        self.db_retriever = Retriever(db_emb, self.db_df)
    
    def embed_query(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=text
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return EmbeddingUtils.sanitize_vec(vec)
    
    def answer(self, query: str) -> Tuple[str, List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Generate answer and return results for display."""
        query_vec = self.embed_query(query)
        
        faq_results = self.faq_retriever.search(query_vec, k=5)
        db_results = self.db_retriever.search(query_vec, k=5)
        
        best_faq_idx, best_faq_score = faq_results[0]
        best_db_idx, best_db_score = db_results[0]
        
        is_db_intent = IntentDetector.is_database_intent(query)
        
        show_faq = best_faq_score >= Config.FAQ_MIN_CONFIDENCE
        show_db = is_db_intent or best_db_score >= Config.DB_MIN_CONFIDENCE
        
        if show_faq and best_faq_score >= Config.FAQ_HIGH_CONFIDENCE and not show_db:
            answer = self._format_faq_answer(best_faq_idx)
            return answer, faq_results[:1], []
        
        if (show_faq and show_db and 
            (best_faq_score - Config.FAQ_MIN_CONFIDENCE) < Config.BOTH_DELTA):
            answer = self._format_both(best_faq_idx, db_results[:3])
            return answer, faq_results[:1], db_results[:3]
        
        if show_db and (is_db_intent or best_db_score > best_faq_score):
            answer = self._format_db_recommendations(db_results[:5])
            return answer, [], db_results[:5]
        
        if show_faq:
            answer = self._format_faq_answer(best_faq_idx)
            return answer, faq_results[:1], []
        
        answer = self._format_unclear()
        return answer, [], []
    
    def _format_faq_answer(self, idx: int) -> str:
        answer = self.faq_df.loc[idx, "answer"]
        return f"📖 **FAQ Answer**\n\n{answer}"
    
    def _format_db_recommendations(self, results: List[Tuple[int, float]]) -> str:
        lines = ["🔎 **Recommended Databases**\n"]
        
        for idx, score in results:
            name = self.db_df.loc[idx, "name"]
            desc = EmbeddingUtils.clean_html(self.db_df.loc[idx, "description"])
            
            if len(desc) > 200:
                desc = desc[:197] + "..."
            
            lines.append(f"**{name}** (confidence: {score:.2f})")
            lines.append(f"{desc}\n")
        
        return "\n".join(lines)
    
    def _format_both(self, faq_idx: int, db_results: List[Tuple[int, float]]) -> str:
        faq_part = self._format_faq_answer(faq_idx)
        db_part = self._format_db_recommendations(db_results)
        return f"{faq_part}\n\n---\n\n{db_part}"
    
    def _format_unclear(self) -> str:
        return (
            "🤔 **I'm not quite sure how to help.** You can:\n\n"
            "- Ask about library services (hours, borrowing, access, etc.)\n"
            "- Request database recommendations for your research topic\n\n"
            "**Example:** *'Which databases should I use for engineering articles?'*"
        )

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # API Key input
    # Try to get from secrets first (for Streamlit Cloud)
    default_api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, 'secrets') else ""
    
    api_key = st.text_input(
        "OpenAI API Key",
        value=default_api_key,
        type="password",
        help="Enter your OpenAI API key to use the chatbot"
    )
    
    if api_key and not st.session_state.api_key_set:
        try:
            with st.spinner("Initializing chatbot..."):
                st.session_state.chatbot = LibraryChatbot(api_key)
                st.session_state.api_key_set = True
            st.success("✅ Chatbot initialized!")
        except Exception as e:
            st.error(f"❌ Error initializing: {e}")
    
    st.divider()
    
    # Statistics
    if st.session_state.chatbot:
        st.subheader("📊 Statistics")
        st.metric("Total FAQs", len(st.session_state.chatbot.faq_df))
        st.metric("Total Databases", len(st.session_state.chatbot.db_df))
        st.metric("Messages Sent", len([m for m in st.session_state.messages if m['role'] == 'user']))
    
    st.divider()
    
    # Threshold controls
    with st.expander("🎛️ Advanced Settings"):
        Config.FAQ_MIN_CONFIDENCE = st.slider(
            "FAQ Confidence Threshold",
            0.0, 1.0, 0.60, 0.05,
            help="Minimum confidence for FAQ answers"
        )
        Config.DB_MIN_CONFIDENCE = st.slider(
            "Database Confidence Threshold",
            0.0, 1.0, 0.45, 0.05,
            help="Minimum confidence for database recommendations"
        )
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Info
    st.markdown("""
    ### 💡 Tips
    - Ask about library policies and services
    - Request database recommendations for your research
    - Be specific about your research topic
    """)

# Main content
st.title("📚 Library Chatbot")
st.markdown("Ask me about library services or get personalized database recommendations!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
if not st.session_state.api_key_set:
    st.info("👈 Please enter your OpenAI API key in the sidebar to start chatting.")
else:
    if prompt := st.chat_input("Ask about library services or database recommendations..."):
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, faq_res, db_res = st.session_state.chatbot.answer(prompt)
                    st.markdown(answer)
                    
                    # Store assistant message
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': answer
                    })
                    
                    # Show debug info in expander
                    with st.expander("🔍 Debug Info"):
                        if faq_res:
                            st.write("**FAQ Results:**")
                            for idx, score in faq_res:
                                st.write(f"- {st.session_state.chatbot.faq_df.loc[idx, 'question'][:100]}... ({score:.3f})")
                        
                        if db_res:
                            st.write("**Database Results:**")
                            for idx, score in db_res:
                                st.write(f"- {st.session_state.chatbot.db_df.loc[idx, 'name']} ({score:.3f})")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    logger.error(f"Error processing query: {e}", exc_info=True)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Made with ❤️ using Streamlit | Powered by OpenAI
    </div>
    """,
    unsafe_allow_html=True
)