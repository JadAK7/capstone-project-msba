import React from 'react';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';

function DebugPanel({ debug, detectedLanguage }) {
  const { language } = useLanguage();

  if (!debug) return null;

  const sourceTypeLabels = {
    faculty_text: 'Faculty Text (custom notes)',
    scraped_website: 'Scraped Website',
    faculty_faq: 'Faculty FAQ',
    databases: 'Databases',
  };

  const sourceTypeColors = {
    faculty_text: '#840132',
    scraped_website: '#1a6b3c',
    faculty_faq: '#2c5282',
    databases: '#7b341e',
  };

  const bestBySource = debug.best_by_source || {};
  const hitsBySource = debug.hits_by_source || {};
  const chunks = debug.retrieved_chunks || [];
  const selectionReason = debug.source_selection_reason || '';

  return (
    <details className="debug-panel">
      <summary className="debug-summary">{t(language, 'debugTitle')}</summary>
      <div className="debug-content">

        {/* Chosen source */}
        <div className="debug-row" style={{ fontSize: '1.05em', fontWeight: 600, marginBottom: 8 }}>
          <strong>Chosen Source:</strong>{' '}
          <span style={{ color: '#840132' }}>{debug.chosen_source || 'N/A'}</span>
        </div>

        {/* Selection reason */}
        {selectionReason && (
          <div className="debug-row" style={{ fontSize: '0.85em', color: '#555', marginBottom: 10, lineHeight: 1.4 }}>
            {selectionReason}
          </div>
        )}

        {/* Best score per source type */}
        {Object.keys(bestBySource).length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <strong style={{ display: 'block', marginBottom: 4 }}>Best Score per Source:</strong>
            <table style={{ width: '100%', fontSize: '0.85em', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #ddd', textAlign: 'left' }}>
                  <th style={{ padding: '4px 8px' }}>Source Type</th>
                  <th style={{ padding: '4px 8px' }}>Best Score</th>
                  <th style={{ padding: '4px 8px' }}>Hits</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(bestBySource)
                  .sort((a, b) => (b[1].score || 0) - (a[1].score || 0))
                  .map(([src, info]) => (
                    <tr key={src} style={{ borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '4px 8px', color: sourceTypeColors[src] || '#333', fontWeight: 500 }}>
                        {sourceTypeLabels[src] || src}
                      </td>
                      <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>
                        {(info.score || 0).toFixed(4)}
                      </td>
                      <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>
                        {(hitsBySource[src] || []).length}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Top hits per source type */}
        {Object.keys(hitsBySource).length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <strong style={{ display: 'block', marginBottom: 6 }}>Top Hits per Source:</strong>
            {Object.entries(hitsBySource)
              .sort((a, b) => {
                const aMax = a[1].length > 0 ? a[1][0].score : 0;
                const bMax = b[1].length > 0 ? b[1][0].score : 0;
                return bMax - aMax;
              })
              .map(([src, hits]) => (
                <details key={src} style={{ marginBottom: 6, borderLeft: `3px solid ${sourceTypeColors[src] || '#ccc'}`, paddingLeft: 8 }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 500, fontSize: '0.9em', color: sourceTypeColors[src] || '#333' }}>
                    {sourceTypeLabels[src] || src} ({hits.length} hit{hits.length !== 1 ? 's' : ''})
                    {hits.length > 0 && <span style={{ fontFamily: 'monospace', marginLeft: 8, color: '#555' }}>best: {hits[0].score.toFixed(3)}</span>}
                  </summary>
                  <div style={{ maxHeight: 200, overflowY: 'auto', marginTop: 4 }}>
                    <table style={{ width: '100%', fontSize: '0.8em', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid #ddd', textAlign: 'left' }}>
                          <th style={{ padding: '3px 6px' }}>Final</th>
                          <th style={{ padding: '3px 6px' }}>Raw</th>
                          <th style={{ padding: '3px 6px' }}>Boost</th>
                          <th style={{ padding: '3px 6px' }}>Vec</th>
                          <th style={{ padding: '3px 6px' }}>Title / Preview</th>
                        </tr>
                      </thead>
                      <tbody>
                        {hits.map((hit, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid #f0f0f0', verticalAlign: 'top' }}>
                            <td style={{ padding: '3px 6px', fontFamily: 'monospace', fontWeight: 600 }}>
                              {hit.score.toFixed(3)}
                            </td>
                            <td style={{ padding: '3px 6px', fontFamily: 'monospace' }}>
                              {hit.raw_score.toFixed(3)}
                            </td>
                            <td style={{ padding: '3px 6px', fontFamily: 'monospace', color: '#888' }}>
                              +{hit.boost.toFixed(3)}
                            </td>
                            <td style={{ padding: '3px 6px', fontFamily: 'monospace', color: '#888' }}>
                              {hit.vector_score.toFixed(3)}
                            </td>
                            <td style={{ padding: '3px 6px', maxWidth: 300 }}>
                              <div style={{ fontWeight: 500 }}>{hit.title || '—'}</div>
                              <div style={{ fontSize: '0.85em', color: '#777', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                {hit.text_preview}
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              ))}
          </div>
        )}

        {/* Retrieved chunks (final selection sent to LLM) */}
        {chunks.length > 0 && (
          <details style={{ marginBottom: 12 }}>
            <summary style={{ cursor: 'pointer', fontWeight: 600, fontSize: '0.9em' }}>
              Chunks Sent to LLM ({chunks.length})
            </summary>
            <div style={{ maxHeight: 250, overflowY: 'auto', marginTop: 4 }}>
              <table style={{ width: '100%', fontSize: '0.8em', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #ddd', textAlign: 'left', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ padding: '4px 6px' }}>#</th>
                    <th style={{ padding: '4px 6px' }}>Source</th>
                    <th style={{ padding: '4px 6px' }}>Score</th>
                    <th style={{ padding: '4px 6px' }}>Title</th>
                  </tr>
                </thead>
                <tbody>
                  {chunks.map((chunk, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #eee', verticalAlign: 'top' }}>
                      <td style={{ padding: '4px 6px', fontFamily: 'monospace' }}>{i + 1}</td>
                      <td style={{ padding: '4px 6px', whiteSpace: 'nowrap', color: sourceTypeColors[chunk.source_type] || '#333' }}>
                        {sourceTypeLabels[chunk.source_type] || chunk.source_type || chunk.source}
                      </td>
                      <td style={{ padding: '4px 6px', fontFamily: 'monospace', fontWeight: 600 }}>
                        {(chunk.score || 0).toFixed(3)}
                      </td>
                      <td style={{ padding: '4px 6px' }}>
                        {chunk.page_title || chunk.text?.slice(0, 80) || '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}

        {/* Other debug info */}
        <div className="debug-row">
          <strong>Pipeline:</strong> {debug.pipeline || 'N/A'}
        </div>
        <div className="debug-row">
          <strong>Cache Hit:</strong> {debug.cache_hit ? 'Yes' : 'No'}
        </div>
        <div className="debug-row">
          <strong>Intent:</strong> {debug.query_intent || 'N/A'}
          {debug.is_db_intent && ' (DB intent)'}
        </div>
        {debug.context_confidence && (
          <div className="debug-row">
            <strong>Confidence:</strong> {debug.context_confidence}
          </div>
        )}
        {debug.search_query && debug.search_query !== debug.query && (
          <div className="debug-row">
            <strong>Rewritten Query:</strong> {debug.search_query}
          </div>
        )}
        {detectedLanguage && (
          <div className="debug-row">
            <strong>{t(language, 'debugDetectedLang')}</strong> {detectedLanguage}
          </div>
        )}
        <div className="debug-row">
          <strong>Candidates:</strong> {debug.total_candidates || 0} retrieved, {debug.reranked_count || 0} after rerank
        </div>
      </div>
    </details>
  );
}

export default DebugPanel;
