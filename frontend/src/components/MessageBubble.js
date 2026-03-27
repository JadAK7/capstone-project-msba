import React from 'react';
import ReactMarkdown from 'react-markdown';
import DebugPanel from './DebugPanel';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';

/**
 * Lightweight client-side Arabic detection for fallback.
 * Used when detectedLanguage is not available on the message.
 */
function detectMessageLanguage(text) {
  const arabicPattern = /[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]/;
  return arabicPattern.test(text) ? 'ar' : 'en';
}

function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  const { language } = useLanguage();

  // Per-message language: use the stored detectedLanguage if available,
  // otherwise detect from the message content. This ensures each bubble
  // renders with correct directionality independent of the global UI language.
  const messageLang = message.detectedLanguage || detectMessageLanguage(message.content);
  const messageDir = messageLang === 'ar' ? 'rtl' : 'ltr';

  return (
    <div className={`message-container ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div
        className={`message-bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}
        dir={messageDir}
        lang={messageLang}
      >
        <div className="message-role">
          {isUser ? t(language, 'roleUser') : t(language, 'roleAssistant')}
        </div>
        <div className="message-content" dir="auto">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
      </div>
      {!isUser && message.debug && (
        <DebugPanel debug={message.debug} detectedLanguage={message.detectedLanguage} />
      )}
    </div>
  );
}

export default MessageBubble;
