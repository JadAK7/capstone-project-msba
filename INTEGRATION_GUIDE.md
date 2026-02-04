# Integration Guide: Adding Library Pages to RAG System

This guide shows you how to integrate the scraped AUB library pages into your chatbot after running `scrape_aub_library.py`.

## Step 1: Run the Scraper

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Run the scraper
python scrape_aub_library.py
```

**Expected Output:**
- `library_pages_emb.npy` - Embeddings for all scraped pages
- `library_pages_text.parquet` - Text data (url, title, content)

## Step 2: Update Configuration

### Option A: Modify `app.py` (Streamlit UI)

**Add library paths to Config class** (around line 39):

```python
class Config:
    # Source data paths
    FAQ_SOURCE = "library_faq_clean.csv"
    DB_SOURCE = "Databases description.csv"

    # Generated index paths
    FAQ_TEXT_PATH = "faq_text.parquet"
    FAQ_EMB_PATH = "faq_emb.npy"
    DB_TEXT_PATH = "db_text.parquet"
    DB_EMB_PATH = "db_emb.npy"

    # ADD THESE TWO LINES:
    LIBRARY_TEXT_PATH = "library_pages_text.parquet"
    LIBRARY_EMB_PATH = "library_pages_emb.npy"

    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE = 200

    FAQ_HIGH_CONFIDENCE = 0.70
    FAQ_MIN_CONFIDENCE = 0.60
    DB_MIN_CONFIDENCE = 0.45
    BOTH_DELTA = 0.06

    # ADD THIS LINE:
    LIBRARY_MIN_CONFIDENCE = 0.50  # Threshold for library page matches
```

**Load library pages in Chatbot.__init__()** (after line 213):

```python
def __init__(self):
    self.client = OpenAI()

    # Build indices if needed
    if not IndexBuilder.indices_exist():
        with st.spinner("Building indices (first-time setup)..."):
            try:
                IndexBuilder.build_indices(self.client)
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

    # ADD THESE 4 LINES:
    # Load library pages data
    self.library_df = pd.read_parquet(Config.LIBRARY_TEXT_PATH)
    library_emb = np.load(Config.LIBRARY_EMB_PATH)
    self.library_retriever = Retriever(library_emb, self.library_df)
```

**Update answer() method** (around line 223):

```python
def answer(self, query: str) -> Tuple[str, List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Generate answer and return results for display."""
    query_vec = self.embed_query(query)

    faq_results = self.faq_retriever.search(query_vec, k=5)
    db_results = self.db_retriever.search(query_vec, k=5)
    # ADD THIS LINE:
    library_results = self.library_retriever.search(query_vec, k=3)

    best_faq_idx, best_faq_score = faq_results[0]
    best_db_idx, best_db_score = db_results[0]
    # ADD THIS LINE:
    best_library_idx, best_library_score = library_results[0]

    is_db_intent = IntentDetector.is_database_intent(query)

    show_faq = best_faq_score >= Config.FAQ_MIN_CONFIDENCE
    show_db = is_db_intent or best_db_score >= Config.DB_MIN_CONFIDENCE
    # ADD THIS LINE:
    show_library = best_library_score >= Config.LIBRARY_MIN_CONFIDENCE

    # HIGH CONFIDENCE FAQ
    if show_faq and best_faq_score >= Config.FAQ_HIGH_CONFIDENCE and not show_db:
        answer = self._format_faq_answer(best_faq_idx)
        return answer, faq_results[:1], []

    # BOTH FAQ AND DB
    if (show_faq and show_db and
        (best_faq_score - Config.FAQ_MIN_CONFIDENCE) < Config.BOTH_DELTA):
        answer = self._format_both(best_faq_idx, db_results[:3])
        return answer, faq_results[:1], db_results[:3]

    # DATABASE RECOMMENDATIONS
    if show_db and (is_db_intent or best_db_score > best_faq_score):
        answer = self._format_db_recommendations(db_results[:5])
        return answer, [], db_results[:5]

    # FAQ ANSWER
    if show_faq:
        answer = self._format_faq_answer(best_faq_idx)
        return answer, faq_results[:1], []

    # ADD THIS BLOCK:
    # LIBRARY PAGES (fallback for general library info)
    if show_library:
        answer = self._format_library_answer(library_results[:3])
        return answer, [], []

    # UNCLEAR
    return self._format_unclear(), [], []
```

**Add formatting method for library answers** (after the other format methods):

```python
def _format_library_answer(self, results: List[Tuple[int, float]]) -> str:
    """Format library page results."""
    lines = ["I found some relevant information from the library website:\n"]

    for i, (idx, score) in enumerate(results, 1):
        row = self.library_df.iloc[idx]
        title = row['title']
        url = row['url']
        content = row['content'][:300]  # First 300 chars

        lines.append(f"**{i}. {title}**")
        lines.append(f"   {content}...")
        lines.append(f"   [Read more]({url})\n")

    return "\n".join(lines)
```

### Option B: Modify `chat.py` (CLI version)

Apply the same changes to `chat.py`:
1. Add library paths to Config class (line 32)
2. Load library retriever in __init__ (after line 159)
3. Update answer() method to include library results
4. Add _format_library_answer() method

## Step 3: Test the Integration

### Test in Streamlit:
```bash
streamlit run app.py
```

### Test in CLI:
```bash
python chat.py
```

### Example Queries to Test:

1. **FAQ Query**: "How do I reserve a book?"
   - Should return FAQ answer

2. **Database Query**: "Which database should I use for engineering research?"
   - Should return database recommendations

3. **Library Info Query**: "What are the library hours?"
   - Should return library page content

4. **General Library Query**: "Tell me about library services"
   - Should return relevant library pages

## Step 4: Verify the Data

Check that the files were created:
```bash
ls -lh library_pages_*.{npy,parquet}
```

Preview the scraped data:
```python
import pandas as pd
import numpy as np

# Load and check
df = pd.read_parquet("library_pages_text.parquet")
emb = np.load("library_pages_emb.npy")

print(f"Scraped pages: {len(df)}")
print(f"Embeddings shape: {emb.shape}")
print("\nSample pages:")
print(df[['title', 'url']].head())
```

## Step 5: Fine-tuning (Optional)

### Adjust Confidence Thresholds

If library pages are showing up too often or not enough:

```python
# In Config class
LIBRARY_MIN_CONFIDENCE = 0.50  # Increase to be more selective
LIBRARY_MIN_CONFIDENCE = 0.40  # Decrease to show more results
```

### Adjust Response Priority

You can change when library pages appear by reordering the logic in the `answer()` method. Current priority:
1. High confidence FAQ
2. FAQ + Database (both)
3. Database recommendations
4. FAQ answer
5. **Library pages** (fallback)
6. Unclear

### Add Library Page Previews to Streamlit

To show library pages in the sidebar like FAQ/DB results, update the display logic:

```python
# In the main Streamlit section
if library_results:
    with st.expander("📄 Library Pages", expanded=True):
        for idx, score in library_results:
            row = st.session_state.chatbot.library_df.iloc[idx]
            st.markdown(f"**{row['title']}** (score: {score:.2f})")
            st.markdown(f"[View page]({row['url']})")
            st.divider()
```

## Troubleshooting

### Issue: File not found error
**Solution**: Make sure `library_pages_emb.npy` and `library_pages_text.parquet` exist in the same directory as your code.

### Issue: No library pages showing up
**Solution**: Lower the `LIBRARY_MIN_CONFIDENCE` threshold or check that the scraper actually found pages.

### Issue: Too many library pages showing up
**Solution**: Increase the `LIBRARY_MIN_CONFIDENCE` threshold to be more selective.

### Issue: Import errors
**Solution**: Run `pip install -r requirements.txt` to ensure all dependencies are installed.

## Architecture After Integration

```
User Query
    ↓
Embed Query (OpenAI)
    ↓
    ├──→ Search FAQ Index (43 Q&As)
    ├──→ Search Database Index (422 databases)
    └──→ Search Library Pages Index (N scraped pages)  ← NEW!
    ↓
Intent Detection + Scoring
    ↓
Response Router
    ├──→ FAQ Answer
    ├──→ Database Recommendations
    ├──→ Library Pages Info  ← NEW!
    └──→ Unclear/Clarification
```

## Summary

After running the scraper:
1. ✅ Two files will be created: `library_pages_emb.npy` and `library_pages_text.parquet`
2. ✅ Add library paths to `Config` class
3. ✅ Load library retriever in `__init__`
4. ✅ Add library search to `answer()` method
5. ✅ Create `_format_library_answer()` method
6. ✅ Test with various queries
7. ✅ Adjust thresholds as needed

Your RAG system will now have three sources of information:
- **FAQ** (specific library questions)
- **Databases** (research database recommendations)
- **Library Pages** (general library information from website)
