import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import AdminDashboard from './pages/AdminDashboard';
import { sendMessage, checkHealth, submitPublicFeedback } from './api';
import { LanguageProvider, useLanguage } from './LanguageContext';
import { t } from './i18n';

/** Maximum number of conversation turns (user+assistant pairs) to send as history. */
const MAX_HISTORY_TURNS = 5;

/**
 * Lightweight client-side Arabic detection.
 * Mirrors the backend's LanguageDetector logic: any Arabic Unicode character
 * present means the message is in Arabic.
 */
function detectLanguage(text) {
  const arabicPattern = /[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]/;
  return arabicPattern.test(text) ? 'ar' : 'en';
}

function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const { language, isRTL } = useLanguage();

  useEffect(() => {
    checkHealth()
      .then(() => {
        setBackendStatus('connected');
      })
      .catch((err) => {
        setBackendStatus('error');
        setError(`Backend connection failed: ${err.message}`);
      });
  }, []);

  const handleSubmit = async (message) => {
    // Detect language on the user message for per-message RTL/LTR rendering
    const userMessage = {
      role: 'user',
      content: message,
      detectedLanguage: detectLanguage(message),
    };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setError(null);

    // Build history from existing messages (before this new message).
    // messages state captures all turns prior to the current user message.
    const maxEntries = MAX_HISTORY_TURNS * 2;
    const recentMessages = messages.slice(-maxEntries);
    const history = recentMessages.map(({ role, content }) => ({ role, content }));

    try {
      // Never send a language parameter -- the backend auto-detects per message
      const response = await sendMessage(message, history);
      const detectedLang = response.detected_language || null;
      const assistantMessage = {
        role: 'assistant',
        content: response.answer || 'No response received.',
        debug: response.debug || null,
        detectedLanguage: detectedLang,
        conversationId: response.conversation_id || null,
        feedbackGiven: null,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      let userMsg;
      if (err.message === 'TIMEOUT') {
        userMsg = 'The request timed out. The server may be busy — please try again.';
      } else if (err.message === 'SERVICE_UNAVAILABLE') {
        userMsg = 'The service is temporarily unavailable. Please try again in a moment.';
      } else {
        userMsg = `Sorry, I encountered an error. Please try again.`;
      }
      setError(userMsg);
      const errorMessage = {
        role: 'assistant',
        content: userMsg,
        detectedLanguage: 'en',
        isError: true,
        retryMessage: message,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (messageIndex, rating, userComment) => {
    const msg = messages[messageIndex];
    if (!msg || !msg.conversationId) return;
    try {
      await submitPublicFeedback(
        msg.conversationId,
        rating,
        userComment || null,
      );
      setMessages((prev) =>
        prev.map((m, i) => i === messageIndex ? { ...m, feedbackGiven: rating } : m)
      );
    } catch (err) {
      console.error('Feedback submission failed:', err);
    }
  };

  return (
    <div className="app-container" dir={isRTL ? 'rtl' : 'ltr'} lang={language}>
      <Header />

      <div className="subtitle-banner">
        <p>
          <strong>{t(language, 'subtitleBanner.prefix')}</strong>{t(language, 'subtitleBanner.suffix')}
        </p>
      </div>

      {backendStatus === 'checking' && (
        <div className="status-banner status-info">
          {t(language, 'statusConnecting')}
        </div>
      )}

      {backendStatus === 'error' && (
        <div className="status-banner status-error">
          {error}
        </div>
      )}

      <main className="main-content">
        <ChatWindow messages={messages} loading={loading} onFeedback={handleFeedback} />
        <ChatInput onSubmit={handleSubmit} disabled={loading || backendStatus === 'error'} />
      </main>

      <Footer />
    </div>
  );
}

function App() {
  return (
    <LanguageProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/admin" element={<AdminDashboard />} />
        </Routes>
      </BrowserRouter>
    </LanguageProvider>
  );
}

export default App;
