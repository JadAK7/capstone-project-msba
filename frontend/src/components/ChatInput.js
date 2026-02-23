import React, { useState } from 'react';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';

function ChatInput({ onSubmit, disabled }) {
  const [input, setInput] = useState('');
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
    <form className="chat-input-form" onSubmit={handleSubmit}>
      <div className="chat-input-container">
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
  );
}

export default ChatInput;
