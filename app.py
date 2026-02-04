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
    layout="centered",
    initial_sidebar_state="collapsed"
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
    LIBRARY_TEXT_PATH = "library_pages_text.parquet"
    LIBRARY_EMB_PATH = "library_pages_emb.npy"

    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE = 200

    FAQ_HIGH_CONFIDENCE = 0.70
    FAQ_MIN_CONFIDENCE = 0.60
    DB_MIN_CONFIDENCE = 0.45
    LIBRARY_MIN_CONFIDENCE = 0.35
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

    @staticmethod
    def clean_library_content(s: str) -> str:
        """Strip AUB navigation boilerplate from scraped page content."""
        # Strip toolbar + breadcrumbs up to the first title-case word
        s = re.sub(
            r"^.*?HOME\s*>\s*LIBRARIES[^a-z]*(?=[A-Z][a-z])",
            "", s, count=1, flags=re.DOTALL
        )
        # Strip trailing sidebar navigation that appears after main content
        s = re.sub(
            r"\s+SERVICES\s+DIRECTIONS & ACCESSIBILITY\s+FOR ALUMNI.*$",
            "", s, flags=re.DOTALL
        )
        # Clean up leftover whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

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

        # Load library pages data
        try:
            self.library_df = pd.read_parquet(Config.LIBRARY_TEXT_PATH)
            library_emb = np.load(Config.LIBRARY_EMB_PATH)
            self.library_retriever = Retriever(library_emb, self.library_df)
        except FileNotFoundError:
            logger.warning("Library pages not found. Run scrape_aub_library.py to create them.")
            self.library_df = pd.DataFrame()
            self.library_retriever = None
    
    def embed_query(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=text
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return EmbeddingUtils.sanitize_vec(vec)
    
    def answer(self, query: str) -> Tuple[str, dict]:
        """Generate answer and return results with debug info."""
        query_vec = self.embed_query(query)

        faq_results = self.faq_retriever.search(query_vec, k=5)
        db_results = self.db_retriever.search(query_vec, k=5)

        # Search library pages if available
        library_results = []
        if self.library_retriever is not None:
            library_results = self.library_retriever.search(query_vec, k=3)

        best_faq_idx, best_faq_score = faq_results[0]
        best_db_idx, best_db_score = db_results[0]
        best_library_score = library_results[0][1] if library_results else 0.0

        is_db_intent = IntentDetector.is_database_intent(query)

        show_faq = best_faq_score >= Config.FAQ_MIN_CONFIDENCE
        show_db = is_db_intent or best_db_score >= Config.DB_MIN_CONFIDENCE
        show_library = bool(library_results) and best_library_score >= Config.LIBRARY_MIN_CONFIDENCE

        debug = {
            "faq_results": faq_results,
            "db_results": db_results,
            "library_results": library_results,
            "is_db_intent": is_db_intent,
            "show_faq": show_faq,
            "show_db": show_db,
            "show_library": show_library,
            "library_available": self.library_retriever is not None,
            "chosen_source": None,
        }

        # 1. Database intent → always use database recommendations
        if show_db and is_db_intent:
            debug["chosen_source"] = "database (keyword intent)"
            return self._format_db_recommendations(db_results[:5]), debug

        # 2. Scraped library pages → primary source for non-DB questions
        if show_library:
            debug["chosen_source"] = "library pages (scraped)"
            return self._format_library_answer(query, library_results[:3]), debug

        # 3. FAQ → backup if scraped data didn't match
        if show_faq:
            debug["chosen_source"] = "FAQ"
            return self._format_faq_answer(best_faq_idx), debug

        # 4. Database recommendations by semantic score (no keyword intent)
        if show_db:
            debug["chosen_source"] = "database (semantic)"
            return self._format_db_recommendations(db_results[:5]), debug

        # 5. Nothing matched
        debug["chosen_source"] = "none (unclear)"
        return self._format_unclear(), debug
    
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

    def _format_library_answer(self, query: str, results: List[Tuple[int, float]]) -> str:
        """Use the LLM to synthesize a clean answer from retrieved library pages."""
        # Build context from retrieved pages
        context_parts = []
        sources = []
        for idx, score in results:
            row = self.library_df.iloc[idx]
            content = EmbeddingUtils.clean_library_content(row['content'])
            context_parts.append(f"Page: {row['title']}\n{content}")
            sources.append(f"[{row['title']}]({row['url']})")

        context = "\n\n---\n\n".join(context_parts)

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AUB library assistant. Answer the student's question "
                        "using ONLY the provided context from the library website. "
                        "Be concise and directly answer what was asked. "
                        "If the context doesn't contain the answer, say so. "
                        "Use markdown formatting for readability."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            temperature=0.2,
            max_tokens=500,
        )

        answer = resp.choices[0].message.content
        source_links = " | ".join(sources)
        return f"📄 {answer}\n\n**Sources:** {source_links}"

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

# Initialize API key from secrets only (no user input)
if not st.session_state.api_key_set:
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        
        if not api_key:
            st.error("⚠️ API key not configured. Please contact the administrator.")
            st.stop()
        
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = LibraryChatbot(api_key)
            st.session_state.api_key_set = True
            
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {e}")
        st.stop()

# Main content
st.title("📚 Library Chatbot")
st.markdown("Ask me about library services or get personalized database recommendations!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
if prompt := st.chat_input("Ask about library services or database recommendations..."):
    # Add user message
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, debug = st.session_state.chatbot.answer(prompt)
                st.markdown(answer)

                # Store assistant message
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': answer
                })

                bot = st.session_state.chatbot

                # Show debug info in expander
                with st.expander("🔍 Debug Info"):
                    st.write(f"**Chosen source:** `{debug['chosen_source']}`")
                    st.write(f"**DB keyword intent:** `{debug['is_db_intent']}`")
                    st.write(f"**Library retriever loaded:** `{debug['library_available']}`")
                    st.write("---")

                    st.write(f"**Top Library Page results** (threshold: {Config.LIBRARY_MIN_CONFIDENCE}, pass: `{debug['show_library']}`):")
                    if debug["library_results"]:
                        for idx, score in debug["library_results"]:
                            title = bot.library_df.iloc[idx]['title'][:80]
                            st.write(f"- `{score:.3f}` — {title}")
                    else:
                        st.write("- *(no library data loaded)*")

                    st.write(f"**Top FAQ results** (threshold: {Config.FAQ_MIN_CONFIDENCE}, pass: `{debug['show_faq']}`):")
                    for idx, score in debug["faq_results"][:3]:
                        q = bot.faq_df.loc[idx, 'question'][:80]
                        st.write(f"- `{score:.3f}` — {q}")

                    st.write(f"**Top Database results** (threshold: {Config.DB_MIN_CONFIDENCE}, pass: `{debug['show_db']}`):")
                    for idx, score in debug["db_results"][:3]:
                        name = bot.db_df.loc[idx, 'name']
                        st.write(f"- `{score:.3f}` — {name}")
            
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