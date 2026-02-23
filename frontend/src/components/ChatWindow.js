import React, { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';

function ChatWindow({ messages, loading }) {
  const scrollRef = useRef(null);
  const { language } = useLanguage();

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const exampleQuestions = t(language, 'exampleQuestions');

  return (
    <div className="chat-window" ref={scrollRef}>
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h2>{t(language, 'welcomeTitle')}</h2>
            <p>{t(language, 'welcomeDescription')}</p>
            <div className="example-questions">
              <p><strong>{t(language, 'exampleQuestionsTitle')}</strong></p>
              <ul>
                {exampleQuestions.map((question, index) => (
                  <li key={index}>{question}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
        {messages.map((msg, index) => (
          <MessageBubble key={index} message={msg} />
        ))}
        {loading && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <span className="loading-text">{t(language, 'loadingText')}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatWindow;
