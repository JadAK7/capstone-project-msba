import React, { useState } from 'react';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';
import EscalationModal from './EscalationModal';

function ChatInput({ onSubmit, disabled }) {
  const [input, setInput] = useState('');
  const [showEscalation, setShowEscalation] = useState(false);
  const { language } = useLanguage();

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmedInput = input.trim();
    if (trimmedInput && !disabled) {
      onSubmit(trimmedInput);
      setInput('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <>
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <div className="chat-input-container">
          <button
            type="button"
            className="escalation-trigger-btn"
            onClick={() => setShowEscalation(true)}
            title={t(language, 'escalation.buttonTitle')}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          </button>
          <input
            type="text"
            className="chat-input"
            placeholder={t(language, 'inputPlaceholder')}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={disabled}
            dir="auto"
          />
          <button
            type="submit"
            className="chat-submit-button"
            disabled={disabled || !input.trim()}
          >
            {t(language, 'sendButton')}
          </button>
        </div>
      </form>

      {showEscalation && (
        <EscalationModal onClose={() => setShowEscalation(false)} />
      )}
    </>
  );
}

export default ChatInput;
