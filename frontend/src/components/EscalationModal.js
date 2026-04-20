import React, { useState } from 'react';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';
import { submitEscalation } from '../api';

function EscalationModal({ onClose }) {
  const { language, isRTL } = useLanguage();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [question, setQuestion] = useState('');
  const [sending, setSending] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email.trim() || !question.trim()) return;

    setSending(true);
    setError(null);

    try {
      await submitEscalation(email.trim(), name.trim(), question.trim());
      setSuccess(true);
    } catch (err) {
      setError(t(language, 'escalation.errorMessage'));
    } finally {
      setSending(false);
    }
  };

  if (success) {
    return (
      <div className="escalation-overlay" onClick={onClose}>
        <div
          className="escalation-modal"
          dir={isRTL ? 'rtl' : 'ltr'}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="escalation-success">
            <div className="escalation-success-icon">&#10003;</div>
            <p>{t(language, 'escalation.successMessage')}</p>
            <button className="escalation-btn escalation-btn-primary" onClick={onClose}>
              OK
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="escalation-overlay" onClick={onClose}>
      <div
        className="escalation-modal"
        dir={isRTL ? 'rtl' : 'ltr'}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="escalation-header">
          <h3>{t(language, 'escalation.modalTitle')}</h3>
          <button className="escalation-close" onClick={onClose}>&times;</button>
        </div>

        <p className="escalation-description">
          {t(language, 'escalation.modalDescription')}
        </p>

        <form onSubmit={handleSubmit} className="escalation-form">
          <label className="escalation-label">
            {t(language, 'escalation.nameLabel')}
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t(language, 'escalation.namePlaceholder')}
              className="escalation-input"
              dir="auto"
            />
          </label>

          <label className="escalation-label">
            {t(language, 'escalation.emailLabel')} *
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder={t(language, 'escalation.emailPlaceholder')}
              className="escalation-input"
              required
              dir="ltr"
            />
          </label>

          <label className="escalation-label">
            {t(language, 'escalation.questionLabel')} *
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={t(language, 'escalation.questionPlaceholder')}
              className="escalation-textarea"
              rows={4}
              required
              dir="auto"
            />
          </label>

          {error && <div className="escalation-error">{error}</div>}

          <div className="escalation-actions">
            <button
              type="button"
              className="escalation-btn escalation-btn-secondary"
              onClick={onClose}
              disabled={sending}
            >
              {t(language, 'escalation.cancelButton')}
            </button>
            <button
              type="submit"
              className="escalation-btn escalation-btn-primary"
              disabled={sending || !email.trim() || !question.trim()}
            >
              {sending ? t(language, 'escalation.sending') : t(language, 'escalation.submitButton')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default EscalationModal;
