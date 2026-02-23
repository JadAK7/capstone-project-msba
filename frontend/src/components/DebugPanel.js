import React from 'react';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';

function DebugPanel({ debug, detectedLanguage }) {
  const { language } = useLanguage();

  if (!debug) return null;

  const formatScores = (scores, threshold) => {
    if (!scores || scores.length === 0) return 'None';
    return scores
      .slice(0, 3)
      .map(s => `${s.toFixed(3)} ${threshold && s >= threshold ? '✓' : ''}`)
      .join(', ');
  };

  return (
    <details className="debug-panel">
      <summary className="debug-summary">{t(language, 'debugTitle')}</summary>
      <div className="debug-content">
        <div className="debug-row">
          <strong>{t(language, 'debugChosenSource')}</strong> {debug.chosen_source || 'N/A'}
        </div>
        <div className="debug-row">
          <strong>{t(language, 'debugDbIntent')}</strong> {debug.db_keyword_detected ? t(language, 'yes') : t(language, 'no')}
        </div>
        <div className="debug-row">
          <strong>{t(language, 'debugLibraryAvailable')}</strong> {debug.library_available ? t(language, 'yes') : t(language, 'no')}
        </div>
        {detectedLanguage && (
          <div className="debug-row">
            <strong>{t(language, 'debugDetectedLang')}</strong> {detectedLanguage}
          </div>
        )}
        {debug.top_faq_scores && debug.top_faq_scores.length > 0 && (
          <div className="debug-row">
            <strong>Top FAQ Scores:</strong> {formatScores(debug.top_faq_scores, debug.faq_threshold)}
            {debug.faq_threshold && ` (threshold: ${debug.faq_threshold.toFixed(2)})`}
          </div>
        )}
        {debug.top_db_scores && debug.top_db_scores.length > 0 && (
          <div className="debug-row">
            <strong>Top DB Scores:</strong> {formatScores(debug.top_db_scores, debug.db_threshold)}
            {debug.db_threshold && ` (threshold: ${debug.db_threshold.toFixed(2)})`}
          </div>
        )}
        {debug.top_library_scores && debug.top_library_scores.length > 0 && (
          <div className="debug-row">
            <strong>Top Library Scores:</strong> {formatScores(debug.top_library_scores, debug.library_threshold)}
            {debug.library_threshold && ` (threshold: ${debug.library_threshold.toFixed(2)})`}
          </div>
        )}
      </div>
    </details>
  );
}

export default DebugPanel;
