import React from 'react';
import ReactMarkdown from 'react-markdown';
import DebugPanel from './DebugPanel';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';

function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  const { language } = useLanguage();

  return (
    <div className={`message-container ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className={`message-bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}>
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
