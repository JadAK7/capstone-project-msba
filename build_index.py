"""
build_index.py
Builds embedding indices for FAQ and Database recommendation system.
Improvements:
- Better error handling and logging
- Configuration management
- Progress tracking
- Data validation
- Modular design
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
from typing import List, Tuple
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    FAQ_PATH = "library_faq_clean.csv"
    DB_PATH = "Databases description.csv"
    
    FAQ_TEXT_OUTPUT = "faq_text.parquet"
    FAQ_EMB_OUTPUT = "faq_emb.npy"
    DB_TEXT_OUTPUT = "db_text.parquet"
    DB_EMB_OUTPUT = "db_emb.npy"
    
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE = 200
    EMBEDDING_DIM = 1536  # for text-embedding-3-small

client = OpenAI()

def sanitize_matrix(m: np.ndarray) -> np.ndarray:
    """
    Normalize and sanitize embedding matrix for safe similarity computation.
    
    Args:
        m: Input embedding matrix
        
    Returns:
        Sanitized and L2-normalized matrix
    """
    m = np.asarray(m, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    m = np.clip(m, -10.0, 10.0)
    
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / np.clip(norms, 1e-12, None)
    
    return np.ascontiguousarray(m)

def embed_batch(
    text_list: List[str],
    model: str = Config.EMBEDDING_MODEL,
    batch_size: int = Config.EMBEDDING_BATCH_SIZE
) -> np.ndarray:
    """
    Generate embeddings for a list of texts with batching and progress tracking.
    
    Args:
        text_list: List of text strings to embed
        model: OpenAI embedding model name
        batch_size: Number of texts per API call
        
    Returns:
        Sanitized embedding matrix
    """
    embeddings = []
    
    with tqdm(total=len(text_list), desc="Generating embeddings") as pbar:
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            
            try:
                resp = client.embeddings.create(model=model, input=batch)
                batch_embeddings = [d.embedding for d in resp.data]
                embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
                
            except Exception as e:
                logger.error(f"Error embedding batch {i}-{i+len(batch)}: {e}")
                raise
    
    arr = np.array(embeddings, dtype=np.float32)
    return sanitize_matrix(arr)

def validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    name: str
) -> Tuple[pd.DataFrame, int]:
    """
    Validate and clean dataframe.
    
    Args:
        df: Input dataframe
        required_cols: List of required column names
        name: Name of dataset for logging
        
    Returns:
        Tuple of (cleaned_df, num_dropped_rows)
    """
    initial_len = len(df)
    
    # Check required columns exist
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name}: Missing columns {missing_cols}")
    
    # Clean and validate
    for col in required_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
    
    # Remove rows with empty required fields
    mask = df[required_cols].apply(lambda x: x != "").all(axis=1)
    df_clean = df[mask].reset_index(drop=True)
    
    dropped = initial_len - len(df_clean)
    if dropped > 0:
        logger.warning(f"{name}: Dropped {dropped} rows with empty fields")
    
    return df_clean, dropped

def build_faq_index() -> None:
    """Build and save FAQ embeddings index."""
    logger.info("Building FAQ index...")
    
    # Load and validate
    faq = pd.read_csv(Config.FAQ_PATH)
    faq, dropped = validate_dataframe(faq, ["question", "answer"], "FAQ")
    
    logger.info(f"Processing {len(faq)} FAQ entries")
    
    # Generate embeddings
    faq_emb = embed_batch(faq["question"].tolist())
    
    # Validate embedding dimensions
    expected_dim = Config.EMBEDDING_DIM
    if faq_emb.shape[1] != expected_dim:
        logger.warning(
            f"Unexpected embedding dimension: {faq_emb.shape[1]} "
            f"(expected {expected_dim})"
        )
    
    # Save
    faq[["question", "answer"]].to_parquet(Config.FAQ_TEXT_OUTPUT, index=False)
    np.save(Config.FAQ_EMB_OUTPUT, faq_emb)
    
    logger.info(f"✓ FAQ index saved: {len(faq)} entries, shape {faq_emb.shape}")

def build_database_index() -> None:
    """Build and save database recommendation embeddings index."""
    logger.info("Building Database index...")
    
    # Load and validate
    db = pd.read_csv(Config.DB_PATH)
    db, dropped = validate_dataframe(db, ["name", "description"], "Database")
    
    logger.info(f"Processing {len(db)} database entries")
    
    # Create combined text for embedding
    db_text = (db["name"] + ". " + db["description"]).tolist()
    
    # Generate embeddings
    db_emb = embed_batch(db_text)
    
    # Validate embedding dimensions
    expected_dim = Config.EMBEDDING_DIM
    if db_emb.shape[1] != expected_dim:
        logger.warning(
            f"Unexpected embedding dimension: {db_emb.shape[1]} "
            f"(expected {expected_dim})"
        )
    
    # Save
    db[["name", "description"]].to_parquet(Config.DB_TEXT_OUTPUT, index=False)
    np.save(Config.DB_EMB_OUTPUT, db_emb)
    
    logger.info(f"✓ Database index saved: {len(db)} entries, shape {db_emb.shape}")

def verify_indices() -> None:
    """Verify that all index files were created successfully."""
    required_files = [
        Config.FAQ_TEXT_OUTPUT,
        Config.FAQ_EMB_OUTPUT,
        Config.DB_TEXT_OUTPUT,
        Config.DB_EMB_OUTPUT
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        raise FileNotFoundError(f"Missing output files: {missing}")
    
    logger.info("✓ All index files verified")

def main():
    """Main entry point for building indices."""
    try:
        logger.info("Starting index build process...")
        
        build_faq_index()
        build_database_index()
        verify_indices()
        
        logger.info("✅ Index build completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Index build failed: {e}")
        raise

if __name__ == "__main__":
    main()