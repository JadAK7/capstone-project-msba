import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import DebugPanel from './DebugPanel';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';

function detectMessageLanguage(text) {
  const arabicPattern = /[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]/;
  return arabicPattern.test(text) ? 'ar' : 'en';
}

function MessageBubble({ message, onFeedback }) {
  const isUser = message.role === 'user';
  const { language } = useLanguage();
  const [showForm, setShowForm] = useState(false);
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const messageLang = message.detectedLanguage || detectMessageLanguage(message.content);
  const messageDir = messageLang === 'ar' ? 'rtl' : 'ltr';

  const showFeedback = !isUser && message.conversationId && onFeedback;
  const feedbackGiven = message.feedbackGiven;

  const handleThumbsDown = () => {
    setShowForm(true);
  };

  const handleSubmitComment = async () => {
    if (!comment.trim()) return;
    setSubmitting(true);
    try {
      await onFeedback(-1, comment.trim());
      setShowForm(false);
      setComment('');
    } finally {
      setSubmitting(false);
    }
  };

  const handleSkip = async () => {
    setSubmitting(true);
    try {
      await onFeedback(-1, null);
      setShowForm(false);
      setComment('');
    } finally {
      setSubmitting(false);
    }
  };

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

        {/* Feedback buttons — hidden once feedback is given */}
        {showFeedback && !showForm && feedbackGiven === null && (
          <div className="message-feedback">
            <button
              className="feedback-btn"
              onClick={() => onFeedback(1, null)}
              title="Helpful"
              aria-label="Mark as helpful"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
              </svg>
            </button>
            <button
              className="feedback-btn"
              onClick={handleThumbsDown}
              title="Not helpful"
              aria-label="Mark as not helpful"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
              </svg>
            </button>
          </div>
        )}

        {/* Feedback form — appears after thumbs down */}
        {showForm && feedbackGiven === null && (
          <div className="feedback-form">
            <p className="feedback-form-label">
              Tell us what's wrong with this answer:
            </p>
            <textarea
              className="feedback-form-input"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="e.g. The hours are wrong, the library is actually open until 10 PM..."
              rows={3}
              dir="auto"
            />
            <div className="feedback-form-actions">
              <button
                className="feedback-form-submit"
                onClick={handleSubmitComment}
                disabled={submitting || !comment.trim()}
              >
                {submitting ? 'Sending...' : 'Submit'}
              </button>
              <button
                className="feedback-form-skip"
                onClick={handleSkip}
                disabled={submitting}
              >
                Skip
              </button>
              <button
                className="feedback-form-cancel"
                onClick={() => setShowForm(false)}
                disabled={submitting}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
      {!isUser && message.debug && (
        <DebugPanel debug={message.debug} detectedLanguage={message.detectedLanguage} />
      )}
    </div>
  );
}

export default MessageBubble;
