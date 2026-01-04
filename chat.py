"""
chat.py
Library Chatbot with FAQ answers and database recommendations.
Improvements:
- Better architecture with classes
- Improved intent detection
- Conversation history
- More flexible routing logic
- Better UX with formatted output
"""

import re
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

client = OpenAI()

# Configuration
class Config:
    FAQ_TEXT_PATH = "faq_text.parquet"
    FAQ_EMB_PATH = "faq_emb.npy"
    DB_TEXT_PATH = "db_text.parquet"
    DB_EMB_PATH = "db_emb.npy"
    
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Thresholds
    FAQ_HIGH_CONFIDENCE = 0.70  # Very confident FAQ match
    FAQ_MIN_CONFIDENCE = 0.60   # Minimum for FAQ answer
    DB_MIN_CONFIDENCE = 0.45    # Minimum for DB recommendation
    BOTH_DELTA = 0.06           # Show both if FAQ barely above threshold

class IntentType(Enum):
    FAQ = "faq"
    DATABASE = "database"
    BOTH = "both"
    UNCLEAR = "unclear"

@dataclass
class SearchResult:
    idx: int
    score: float
    text: str

class EmbeddingUtils:
    """Utilities for safe embedding operations."""
    
    @staticmethod
    def sanitize_matrix(m: np.ndarray) -> np.ndarray:
        """Normalize and sanitize embedding matrix."""
        m = np.asarray(m, dtype=np.float32)
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        m = np.clip(m, -10.0, 10.0)
        
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        m = m / np.clip(norms, 1e-12, None)
        
        return np.ascontiguousarray(m)
    
    @staticmethod
    def sanitize_vec(v: np.ndarray) -> np.ndarray:
        """Normalize and sanitize embedding vector."""
        v = np.asarray(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.clip(v, -10.0, 10.0)
        
        n = np.linalg.norm(v)
        v = v / np.clip(n, 1e-12, None)
        
        return np.ascontiguousarray(v)
    
    @staticmethod
    def clean_html(s: str) -> str:
        """Remove HTML tags and clean text."""
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
    def detect(cls, query: str) -> IntentType:
        """Detect primary intent from query text."""
        has_db_keywords = bool(cls.DB_KEYWORDS.search(query))
        has_research_topics = bool(cls.RESEARCH_TOPICS.search(query))
        
        if has_db_keywords or has_research_topics:
            return IntentType.DATABASE
        
        return IntentType.FAQ  # Default to FAQ

class Retriever:
    """Handles embedding-based retrieval."""
    
    def __init__(self, embeddings: np.ndarray, df: pd.DataFrame):
        self.embeddings = EmbeddingUtils.sanitize_matrix(embeddings)
        self.df = df
    
    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for top-k most similar items.
        
        Args:
            query_vec: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        query_vec = EmbeddingUtils.sanitize_vec(query_vec)
        
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            similarities = self.embeddings @ query_vec
        
        similarities = np.nan_to_num(similarities, nan=-1.0, posinf=-1.0, neginf=-1.0)
        similarities = np.clip(similarities, -1.0, 1.0)
        
        top_indices = np.argsort(-similarities)[:k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

class LibraryChatbot:
    """Main chatbot class coordinating FAQ and database recommendations."""
    
    def __init__(self):
        # Load FAQ data
        self.faq_df = pd.read_parquet(Config.FAQ_TEXT_PATH)
        faq_emb = np.load(Config.FAQ_EMB_PATH)
        self.faq_retriever = Retriever(faq_emb, self.faq_df)
        
        # Load database data
        self.db_df = pd.read_parquet(Config.DB_TEXT_PATH)
        db_emb = np.load(Config.DB_EMB_PATH)
        self.db_retriever = Retriever(db_emb, self.db_df)
        
        logger.info(f"Loaded {len(self.faq_df)} FAQs and {len(self.db_df)} databases")
    
    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for query text."""
        resp = client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=text
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return EmbeddingUtils.sanitize_vec(vec)
    
    def answer(self, query: str) -> str:
        """
        Generate answer for user query.
        
        Args:
            query: User's question
            
        Returns:
            Formatted answer string
        """
        # Embed query
        query_vec = self.embed_query(query)
        
        # Search both indices
        faq_results = self.faq_retriever.search(query_vec, k=5)
        db_results = self.db_retriever.search(query_vec, k=5)
        
        best_faq_idx, best_faq_score = faq_results[0]
        best_db_idx, best_db_score = db_results[0]
        
        # Detect intent
        intent = IntentDetector.detect(query)
        
        # Determine response strategy
        show_faq = best_faq_score >= Config.FAQ_MIN_CONFIDENCE
        show_db = (
            intent == IntentType.DATABASE or 
            best_db_score >= Config.DB_MIN_CONFIDENCE
        )
        
        # High confidence FAQ - just answer
        if show_faq and best_faq_score >= Config.FAQ_HIGH_CONFIDENCE and not show_db:
            return self._format_faq_answer(best_faq_idx)
        
        # Both relevant - show both
        if (show_faq and show_db and 
            (best_faq_score - Config.FAQ_MIN_CONFIDENCE) < Config.BOTH_DELTA):
            return self._format_both(best_faq_idx, db_results[:3])
        
        # Database request or DB score much higher
        if show_db and (intent == IntentType.DATABASE or 
                       best_db_score > best_faq_score):
            return self._format_db_recommendations(db_results[:5])
        
        # FAQ answer
        if show_faq:
            return self._format_faq_answer(best_faq_idx)
        
        # Nothing confident enough
        return self._format_unclear()
    
    def _format_faq_answer(self, idx: int) -> str:
        """Format FAQ answer."""
        answer = self.faq_df.loc[idx, "answer"]
        return f"📖 {answer}"
    
    def _format_db_recommendations(self, results: List[Tuple[int, float]]) -> str:
        """Format database recommendations."""
        lines = ["🔎 Recommended databases:\n"]
        
        for idx, score in results:
            name = self.db_df.loc[idx, "name"]
            desc = EmbeddingUtils.clean_html(self.db_df.loc[idx, "description"])
            
            # Truncate long descriptions
            if len(desc) > 200:
                desc = desc[:197] + "..."
            
            lines.append(f"  • {name} (confidence: {score:.2f})")
            lines.append(f"    {desc}\n")
        
        return "\n".join(lines)
    
    def _format_both(self, faq_idx: int, db_results: List[Tuple[int, float]]) -> str:
        """Format combined FAQ answer and database recommendations."""
        faq_part = self._format_faq_answer(faq_idx)
        db_part = self._format_db_recommendations(db_results)
        
        return f"{faq_part}\n\n{db_part}"
    
    def _format_unclear(self) -> str:
        """Format unclear intent message."""
        return (
            "🤔 I'm not quite sure how to help. You can:\n"
            "  • Ask about library services (hours, borrowing, access, etc.)\n"
            "  • Request database recommendations for your research topic\n"
            "\nExample: 'Which databases should I use for engineering articles?'"
        )

def main():
    """Main chat loop."""
    print("=" * 60)
    print("📚 Library Chatbot")
    print("=" * 60)
    print("Ask me about library services or get database recommendations!")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    try:
        bot = LibraryChatbot()
    except Exception as e:
        print(f"❌ Error loading chatbot: {e}")
        return
    
    while True:
        try:
            query = input("\n💬 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["quit", "exit", "bye"]:
                print("\n👋 Thanks for using the Library Chatbot. Goodbye!")
                break
            
            answer = bot.answer(query)
            print(f"\n🤖 Bot:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\n❌ Sorry, I encountered an error. Please try rephrasing your question.")

if __name__ == "__main__":
    main()