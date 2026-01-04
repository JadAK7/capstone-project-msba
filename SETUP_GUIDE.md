# Library Chatbot - Streamlit Setup Guide

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Your data files:
  - `library_faq_clean.csv`
  - `Databases description.csv`

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the Indices

First, make sure your CSV files are in the same directory, then run:

```bash
python build_index.py
```

This will create:
- `faq_text.parquet` and `faq_emb.npy`
- `db_text.parquet` and `db_emb.npy`

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### 4. Configure the App

1. Enter your OpenAI API key in the sidebar
2. Wait for the chatbot to initialize
3. Start asking questions!

## 📁 Project Structure

```
your-project/
├── app.py                          # Streamlit interface
├── build_index.py                  # Index builder
├── chat.py                         # Command-line version (optional)
├── requirements.txt                # Dependencies
├── library_faq_clean.csv          # Your FAQ data
├── Databases description.csv       # Your database data
├── faq_text.parquet               # Generated
├── faq_emb.npy                    # Generated
├── db_text.parquet                # Generated
└── db_emb.npy                     # Generated
```

## 🎨 Features

### Main Features
- 💬 Interactive chat interface
- 📖 FAQ answering
- 🔎 Database recommendations
- 📊 Real-time statistics
- 🎛️ Adjustable confidence thresholds
- 🔍 Debug information for each response

### Sidebar Controls
- API key management
- Statistics display
- Advanced threshold settings
- Chat history clearing
- Usage tips

## 🔧 Customization

### Adjusting Thresholds

In the sidebar "Advanced Settings":
- **FAQ Confidence Threshold**: Higher = more confident FAQ answers required
- **Database Confidence Threshold**: Higher = more confident DB recommendations required

### Modifying Appearance

Edit `app.py`:
- Change `page_icon` in `st.set_page_config()` for different emoji
- Modify CSS in the footer section
- Adjust layout with `layout="wide"` or `layout="centered"`

### Adding Custom Styling

Add a `.streamlit/config.toml` file:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## 🌐 Deployment Options

### Option 1: Streamlit Community Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add your OpenAI API key in "Secrets" section:
   ```toml
   OPENAI_API_KEY = "your-key-here"
   ```
5. Deploy!

**Note:** For secrets, modify `app.py` to read from `st.secrets`:

```python
# In sidebar, replace:
api_key = st.text_input("OpenAI API Key", type="password")

# With:
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    api_key = st.text_input("OpenAI API Key", type="password")
```

### Option 2: Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t library-chatbot .
docker run -p 8501:8501 library-chatbot
```

### Option 3: Local Network

Run with external access:

```bash
streamlit run app.py --server.address=0.0.0.0
```

Access from other devices on your network using: `http://YOUR_IP:8501`

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: faq_text.parquet"
Run the index builder first:
```bash
python build_index.py
```

### "OpenAI API Error"
- Check your API key is correct
- Ensure you have credits in your OpenAI account
- Check your internet connection

### Slow Response Times
- Embeddings are cached after first generation
- Consider using a faster embedding model
- Check your internet speed

## 💡 Tips for Best Results

1. **Be Specific**: "Which database for electrical engineering research papers?" works better than "database?"

2. **Clear Queries**: The chatbot works best with clear, direct questions

3. **Adjust Thresholds**: If getting too many "unclear" responses, lower the confidence thresholds

4. **Check Debug Info**: Use the debug expander to see what the model is matching

## 📊 Monitoring Usage

Track OpenAI API usage at: https://platform.openai.com/usage

Embedding costs (text-embedding-3-small):
- ~$0.02 per 1M tokens
- Average query: ~50 tokens = ~$0.000001

## 🔒 Security Notes

- **Never commit your API key** to version control
- Use environment variables or Streamlit secrets for production
- Add `.env` to `.gitignore`
- Rotate API keys regularly

## 📞 Support

For issues with:
- **Streamlit**: [Streamlit Docs](https://docs.streamlit.io)
- **OpenAI**: [OpenAI Docs](https://platform.openai.com/docs)
- **This Code**: Check the debug info in the app

## 🎯 Next Steps

Consider adding:
- User feedback collection
- Conversation history export
- Multi-language support
- Custom branding
- Analytics dashboard
- Rate limiting for production