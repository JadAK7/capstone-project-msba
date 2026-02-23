import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import AdminDashboard from './pages/AdminDashboard';
import { sendMessage, checkHealth } from './api';
import { LanguageProvider, useLanguage } from './LanguageContext';
import { t } from './i18n';

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
    const userMessage = { role: 'user', content: message };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setError(null);

    try {
      const response = await sendMessage(message, language);
      const assistantMessage = {
        role: 'assistant',
        content: response.answer || 'No response received.',
        debug: response.debug || null,
        detectedLanguage: response.detected_language || null,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err.message);
      const errorMessage = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${err.message}. Please try again.`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
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
        <ChatWindow messages={messages} loading={loading} />
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
