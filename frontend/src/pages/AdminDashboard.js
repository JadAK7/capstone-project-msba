import React, { useState, useEffect, useCallback } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import {
  getCollections,
  getCollectionEntries,
  addFAQ,
  updateFAQ,
  deleteFAQ,
  addDatabase,
  updateDatabase,
  deleteDatabase,
  deleteLibraryPage,
  triggerReindex,
  triggerRescrape,
  getRescrapeStatus,
  getSystemInfo,
  checkHealth,
  getAnalyticsSummary,
  getAnalyticsTrends,
  getTopQueries,
  getUnansweredQueries,
  getAnalyticsCharts,
  getConversations,
  submitFeedback,
  getFeedbackStats,
} from '../api';

// ---------------------------------------------------------------------------
// Toast notification
// ---------------------------------------------------------------------------
function Toast({ message, type, onClose }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 4000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={`admin-toast admin-toast-${type}`}>
      {message}
      <button className="admin-toast-close" onClick={onClose}>x</button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// System Status Tab
// ---------------------------------------------------------------------------
function SystemStatusTab() {
  const [health, setHealth] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [reindexing, setReindexing] = useState(false);
  const [scraping, setScraping] = useState(false);
  const [scrapeMessage, setScrapeMessage] = useState('');
  const [toast, setToast] = useState(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [h, s] = await Promise.all([checkHealth(), getSystemInfo()]);
      setHealth(h);
      setSystemInfo(s);
    } catch (err) {
      setToast({ message: `Failed to load system info: ${err.message}`, type: 'error' });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const handleReindex = async () => {
    if (!window.confirm('This will rebuild all collections from CSV source files. Continue?')) return;
    setReindexing(true);
    try {
      await triggerReindex();
      setToast({ message: 'Re-indexing completed successfully.', type: 'success' });
      await loadData();
    } catch (err) {
      setToast({ message: `Re-index failed: ${err.message}`, type: 'error' });
    } finally {
      setReindexing(false);
    }
  };

  const handleRescrape = async () => {
    if (!window.confirm('This will re-scrape the AUB library website and rebuild the library pages index. This may take several minutes. Continue?')) return;
    setScraping(true);
    setScrapeMessage('Starting scrape...');
    try {
      await triggerRescrape();
      // Poll for status
      const poll = setInterval(async () => {
        try {
          const status = await getRescrapeStatus();
          setScrapeMessage(status.message || 'Scraping...');
          if (!status.running) {
            clearInterval(poll);
            setScraping(false);
            if (status.error) {
              setToast({ message: `Scrape failed: ${status.error}`, type: 'error' });
            } else {
              setToast({ message: status.message || 'Scraping completed successfully.', type: 'success' });
              await loadData();
            }
          }
        } catch {
          clearInterval(poll);
          setScraping(false);
          setToast({ message: 'Lost connection while checking scrape status.', type: 'error' });
        }
      }, 3000);
    } catch (err) {
      setScraping(false);
      setToast({ message: `Scrape failed: ${err.message}`, type: 'error' });
    }
  };

  if (loading) return <div className="admin-loading">Loading system status...</div>;

  const formatUptime = (seconds) => {
    if (!seconds) return 'N/A';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  };

  const formatTimestamp = (ts) => {
    if (!ts) return 'Never';
    return new Date(ts * 1000).toLocaleString();
  };

  return (
    <div className="admin-tab-content">
      {toast && <Toast {...toast} onClose={() => setToast(null)} />}

      <div className="admin-cards-row">
        <div className="admin-card">
          <div className="admin-card-label">Backend Status</div>
          <div className={`admin-status-badge ${health?.status === 'ok' ? 'status-healthy' : 'status-down'}`}>
            {health?.status === 'ok' ? 'Healthy' : 'Down'}
          </div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">Server Uptime</div>
          <div className="admin-card-value">{formatUptime(systemInfo?.server_uptime_seconds)}</div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">Embedding Model</div>
          <div className="admin-card-value admin-card-value-small">{systemInfo?.embedding_model || 'N/A'}</div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">Last Index Build</div>
          <div className="admin-card-value admin-card-value-small">{formatTimestamp(systemInfo?.last_index_build)}</div>
        </div>
      </div>

      <h3 className="admin-section-title">Collection Counts</h3>
      <div className="admin-cards-row">
        {systemInfo?.collections?.map((col) => (
          <div className="admin-card" key={col.name}>
            <div className="admin-card-label">{col.name}</div>
            <div className="admin-card-value">{col.count}</div>
          </div>
        ))}
      </div>

      <h3 className="admin-section-title">Maintenance</h3>
      <div className="admin-maintenance-buttons">
        <button
          className="admin-btn admin-btn-primary"
          onClick={handleReindex}
          disabled={reindexing || scraping}
        >
          {reindexing ? (
            <span className="admin-btn-loading">Re-indexing...</span>
          ) : (
            'Re-index All Collections'
          )}
        </button>
        <button
          className="admin-btn admin-btn-primary"
          onClick={handleRescrape}
          disabled={scraping || reindexing}
        >
          {scraping ? (
            <span className="admin-btn-loading">Scraping...</span>
          ) : (
            'Re-scrape Library Website'
          )}
        </button>
      </div>
      {scraping && scrapeMessage && (
        <div className="admin-scrape-status">{scrapeMessage}</div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Data Management Tab
// ---------------------------------------------------------------------------
function DataManagementTab() {
  const [activeCollection, setActiveCollection] = useState('faq');
  const [entries, setEntries] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);

  const [reindexing, setReindexing] = useState(false);

  // Add/Edit form state
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [formField1, setFormField1] = useState('');
  const [formField2, setFormField2] = useState('');

  const LIMIT = 20;

  const loadEntries = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getCollectionEntries(activeCollection, offset, LIMIT);
      setEntries(data.entries || []);
      setTotal(data.total || 0);
    } catch (err) {
      setToast({ message: `Failed to load entries: ${err.message}`, type: 'error' });
    } finally {
      setLoading(false);
    }
  }, [activeCollection, offset]);

  useEffect(() => {
    setOffset(0);
    setShowAddForm(false);
    setEditingId(null);
  }, [activeCollection]);

  useEffect(() => { loadEntries(); }, [loadEntries]);

  const resetForm = () => {
    setFormField1('');
    setFormField2('');
    setShowAddForm(false);
    setEditingId(null);
  };

  const handleAdd = async () => {
    if (!formField1.trim() || !formField2.trim()) {
      setToast({ message: 'All fields are required.', type: 'error' });
      return;
    }
    try {
      if (activeCollection === 'faq') {
        await addFAQ(formField1, formField2);
      } else if (activeCollection === 'databases') {
        await addDatabase(formField1, formField2);
      }
      setToast({ message: 'Entry added successfully.', type: 'success' });
      resetForm();
      await loadEntries();
    } catch (err) {
      setToast({ message: `Failed to add entry: ${err.message}`, type: 'error' });
    }
  };

  const handleUpdate = async (id) => {
    if (!formField1.trim() || !formField2.trim()) {
      setToast({ message: 'All fields are required.', type: 'error' });
      return;
    }
    try {
      if (activeCollection === 'faq') {
        await updateFAQ(id, formField1, formField2);
      } else if (activeCollection === 'databases') {
        await updateDatabase(id, formField1, formField2);
      }
      setToast({ message: 'Entry updated successfully.', type: 'success' });
      resetForm();
      await loadEntries();
    } catch (err) {
      setToast({ message: `Failed to update entry: ${err.message}`, type: 'error' });
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm(`Delete entry "${id}"? This cannot be undone.`)) return;
    try {
      if (activeCollection === 'faq') {
        await deleteFAQ(id);
      } else if (activeCollection === 'databases') {
        await deleteDatabase(id);
      } else if (activeCollection === 'library_pages') {
        await deleteLibraryPage(id);
      }
      setToast({ message: 'Entry deleted successfully.', type: 'success' });
      await loadEntries();
    } catch (err) {
      setToast({ message: `Failed to delete entry: ${err.message}`, type: 'error' });
    }
  };

  const startEdit = (entry) => {
    setEditingId(entry.id);
    setShowAddForm(false);
    if (activeCollection === 'faq') {
      setFormField1(entry.metadata?.question || '');
      setFormField2(entry.metadata?.answer || '');
    } else if (activeCollection === 'databases') {
      setFormField1(entry.metadata?.name || '');
      setFormField2(entry.metadata?.description || '');
    }
  };

  const handleReindex = async () => {
    if (!window.confirm('This will rebuild all collections from CSV source files and update embeddings. Continue?')) return;
    setReindexing(true);
    try {
      await triggerReindex();
      setToast({ message: 'Re-indexing completed successfully. Embeddings are now up to date.', type: 'success' });
      await loadEntries();
    } catch (err) {
      setToast({ message: `Re-index failed: ${err.message}`, type: 'error' });
    } finally {
      setReindexing(false);
    }
  };

  const totalPages = Math.ceil(total / LIMIT);
  const currentPage = Math.floor(offset / LIMIT) + 1;

  const field1Label = activeCollection === 'faq' ? 'Question' : 'Name';
  const field2Label = activeCollection === 'faq' ? 'Answer' : 'Description';
  const canEdit = activeCollection !== 'library_pages';

  return (
    <div className="admin-tab-content">
      {toast && <Toast {...toast} onClose={() => setToast(null)} />}

      <div className="admin-collection-tabs">
        {['faq', 'databases', 'library_pages'].map((name) => (
          <button
            key={name}
            className={`admin-collection-tab ${activeCollection === name ? 'active' : ''}`}
            onClick={() => setActiveCollection(name)}
          >
            {name === 'faq' ? 'FAQs' : name === 'databases' ? 'Databases' : 'Library Pages'}
          </button>
        ))}
      </div>

      {activeCollection === 'library_pages' && (
        <div className="admin-info-note">
          Library pages are populated automatically by the web scraper. You can delete entries here
          but cannot add or edit them manually. To refresh this data, use the
          "Re-scrape Library Website" button on the System Status tab.
        </div>
      )}

      {canEdit && (
        <div className="admin-actions-bar">
          <button
            className="admin-btn admin-btn-primary"
            onClick={() => { setShowAddForm(!showAddForm); setEditingId(null); setFormField1(''); setFormField2(''); }}
          >
            {showAddForm ? 'Cancel' : `Add New ${activeCollection === 'faq' ? 'FAQ' : 'Database'}`}
          </button>
        </div>
      )}

      {showAddForm && canEdit && (
        <div className="admin-form">
          <div className="admin-form-group">
            <label>{field1Label}</label>
            <textarea
              value={formField1}
              onChange={(e) => setFormField1(e.target.value)}
              rows={2}
              placeholder={`Enter ${field1Label.toLowerCase()}...`}
            />
          </div>
          <div className="admin-form-group">
            <label>{field2Label}</label>
            <textarea
              value={formField2}
              onChange={(e) => setFormField2(e.target.value)}
              rows={3}
              placeholder={`Enter ${field2Label.toLowerCase()}...`}
            />
          </div>
          <button className="admin-btn admin-btn-primary" onClick={handleAdd}>
            Save
          </button>
        </div>
      )}

      {loading ? (
        <div className="admin-loading">Loading entries...</div>
      ) : (
        <>
          <div className="admin-table-info">
            Showing {entries.length} of {total} entries (page {currentPage} of {totalPages || 1})
          </div>
          <div className="admin-table-wrapper">
            <table className="admin-table">
              <thead>
                <tr>
                  <th>ID</th>
                  {activeCollection === 'faq' && <><th>Question</th><th>Answer</th></>}
                  {activeCollection === 'databases' && <><th>Name</th><th>Description</th></>}
                  {activeCollection === 'library_pages' && <><th>Title</th><th>URL</th><th>Content Preview</th></>}
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => {
                  const isEditing = editingId === entry.id;
                  return (
                    <tr key={entry.id}>
                      <td className="admin-td-id">{entry.id}</td>

                      {activeCollection === 'faq' && (
                        <>
                          <td>
                            {isEditing ? (
                              <textarea
                                className="admin-inline-edit"
                                value={formField1}
                                onChange={(e) => setFormField1(e.target.value)}
                                rows={2}
                              />
                            ) : (
                              <span className="admin-td-text">{entry.metadata?.question || entry.document}</span>
                            )}
                          </td>
                          <td>
                            {isEditing ? (
                              <textarea
                                className="admin-inline-edit"
                                value={formField2}
                                onChange={(e) => setFormField2(e.target.value)}
                                rows={3}
                              />
                            ) : (
                              <span className="admin-td-text">{entry.metadata?.answer || ''}</span>
                            )}
                          </td>
                        </>
                      )}

                      {activeCollection === 'databases' && (
                        <>
                          <td>
                            {isEditing ? (
                              <textarea
                                className="admin-inline-edit"
                                value={formField1}
                                onChange={(e) => setFormField1(e.target.value)}
                                rows={1}
                              />
                            ) : (
                              <span className="admin-td-text">{entry.metadata?.name || ''}</span>
                            )}
                          </td>
                          <td>
                            {isEditing ? (
                              <textarea
                                className="admin-inline-edit"
                                value={formField2}
                                onChange={(e) => setFormField2(e.target.value)}
                                rows={3}
                              />
                            ) : (
                              <span className="admin-td-text admin-td-truncate">{entry.metadata?.description || ''}</span>
                            )}
                          </td>
                        </>
                      )}

                      {activeCollection === 'library_pages' && (
                        <>
                          <td><span className="admin-td-text">{entry.metadata?.title || ''}</span></td>
                          <td>
                            <a
                              href={entry.metadata?.url || '#'}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="admin-link"
                            >
                              {entry.metadata?.url || ''}
                            </a>
                          </td>
                          <td>
                            <span className="admin-td-content-preview">
                              {entry.metadata?.content
                                ? entry.metadata.content.substring(0, 150) + (entry.metadata.content.length > 150 ? '...' : '')
                                : ''}
                            </span>
                          </td>
                        </>
                      )}

                      <td className="admin-td-actions">
                        {isEditing ? (
                          <>
                            <button className="admin-btn admin-btn-sm admin-btn-primary" onClick={() => handleUpdate(entry.id)}>Save</button>
                            <button className="admin-btn admin-btn-sm admin-btn-secondary" onClick={resetForm}>Cancel</button>
                          </>
                        ) : (
                          <>
                            {canEdit && (
                              <button className="admin-btn admin-btn-sm admin-btn-secondary" onClick={() => startEdit(entry)}>Edit</button>
                            )}
                            <button className="admin-btn admin-btn-sm admin-btn-danger" onClick={() => handleDelete(entry.id)}>Delete</button>
                          </>
                        )}
                      </td>
                    </tr>
                  );
                })}
                {entries.length === 0 && (
                  <tr><td colSpan={activeCollection === 'library_pages' ? 5 : 4} className="admin-empty">No entries found.</td></tr>
                )}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="admin-pagination">
              <button
                className="admin-btn admin-btn-sm admin-btn-secondary"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - LIMIT))}
              >
                Previous
              </button>
              <span className="admin-page-info">Page {currentPage} of {totalPages}</span>
              <button
                className="admin-btn admin-btn-sm admin-btn-secondary"
                disabled={offset + LIMIT >= total}
                onClick={() => setOffset(offset + LIMIT)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}

      <div className="admin-reindex-footer">
        <button
          className="admin-btn admin-btn-primary"
          onClick={handleReindex}
          disabled={reindexing}
        >
          {reindexing ? (
            <span className="admin-btn-loading">Re-indexing...</span>
          ) : (
            'Re-index All Collections'
          )}
        </button>
        <span className="admin-reindex-hint">
          Rebuild embeddings after adding, editing, or deleting entries to update search results.
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Analytics Tab (with matplotlib charts)
// ---------------------------------------------------------------------------

const CHART_SECTIONS = [
  { key: 'usage_volume', label: 'Usage & Volume' },
  { key: 'language', label: 'Language' },
  { key: 'retrieval_quality', label: 'Retrieval Quality' },
  { key: 'intent_routing', label: 'Intent & Routing' },
  { key: 'query_analysis', label: 'Query Analysis' },
  { key: 'performance', label: 'Performance' },
  { key: 'knowledge_gaps', label: 'Knowledge Gaps' },
];

function AnalyticsTab() {
  const [summary, setSummary] = useState(null);
  const [topQueries, setTopQueries] = useState([]);
  const [unanswered, setUnanswered] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [extendedSummary, setExtendedSummary] = useState(null);
  const [activeSection, setActiveSection] = useState('usage_volume');
  const [loading, setLoading] = useState(true);
  const [chartsLoading, setChartsLoading] = useState(false);
  const [toast, setToast] = useState(null);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const [s, q, u, c] = await Promise.all([
          getAnalyticsSummary(),
          getTopQueries(),
          getUnansweredQueries(),
          getAnalyticsCharts(),
        ]);
        setSummary(s);
        setTopQueries(q);
        setUnanswered(u);
        setChartData(c.charts || {});
        setExtendedSummary(c.extended_summary || {});
      } catch (err) {
        setToast({ message: `Failed to load analytics: ${err.message}`, type: 'error' });
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const handleRefreshCharts = async () => {
    setChartsLoading(true);
    try {
      const c = await getAnalyticsCharts();
      setChartData(c.charts || {});
      setExtendedSummary(c.extended_summary || {});
      setToast({ message: 'Charts refreshed.', type: 'success' });
    } catch (err) {
      setToast({ message: `Failed to refresh charts: ${err.message}`, type: 'error' });
    } finally {
      setChartsLoading(false);
    }
  };

  if (loading) return <div className="admin-loading">Loading analytics...</div>;

  const currentCharts = chartData?.[activeSection] || {};

  return (
    <div className="admin-tab-content">
      {toast && <Toast {...toast} onClose={() => setToast(null)} />}

      {/* ---- Quick Summary Cards ---- */}
      <h3 className="admin-section-title">Overview</h3>
      <div className="analytics-summary-cards">
        <div className="admin-card">
          <div className="admin-card-label">Total Queries</div>
          <div className="admin-card-value">{summary?.total_conversations || 0}</div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">Today</div>
          <div className="admin-card-value">{summary?.today || 0}</div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">This Week</div>
          <div className="admin-card-value">{summary?.this_week || 0}</div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">This Month</div>
          <div className="admin-card-value">{summary?.this_month || 0}</div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">Est. Sessions</div>
          <div className="admin-card-value">{extendedSummary?.estimated_sessions || 0}</div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">Avg Queries/Session</div>
          <div className="admin-card-value">{extendedSummary?.avg_queries_per_session || 0}</div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">Avg Response Time</div>
          <div className="admin-card-value admin-card-value-small">
            {extendedSummary?.avg_response_time_ms ? `${extendedSummary.avg_response_time_ms}ms` : 'N/A'}
          </div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">P95 Response Time</div>
          <div className="admin-card-value admin-card-value-small">
            {extendedSummary?.p95_response_time_ms ? `${extendedSummary.p95_response_time_ms}ms` : 'N/A'}
          </div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">Cache Hit Rate</div>
          <div className="admin-card-value">{extendedSummary?.cache_hit_rate || 0}%</div>
        </div>
        <div className="admin-card">
          <div className="admin-card-label">Unanswered Rate</div>
          <div className="admin-card-value">{extendedSummary?.unanswered_rate || 0}%</div>
        </div>
      </div>

      {/* ---- Charts Section ---- */}
      <div className="analytics-charts-header">
        <h3 className="admin-section-title" style={{ marginBottom: 0 }}>Visualizations</h3>
        <button
          className="admin-btn admin-btn-sm admin-btn-secondary"
          onClick={handleRefreshCharts}
          disabled={chartsLoading}
        >
          {chartsLoading ? 'Refreshing...' : 'Refresh Charts'}
        </button>
      </div>

      <div className="analytics-section-tabs">
        {CHART_SECTIONS.map((s) => (
          <button
            key={s.key}
            className={`analytics-section-tab ${activeSection === s.key ? 'active' : ''}`}
            onClick={() => setActiveSection(s.key)}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div className="analytics-charts-grid">
        {Object.entries(currentCharts).map(([key, chart]) => (
          <div className="analytics-chart-card" key={key}>
            <h4 className="analytics-chart-title">{chart?.title || key}</h4>
            {chart?.image ? (
              <img
                src={`data:image/png;base64,${chart.image}`}
                alt={chart.title || key}
                className="analytics-chart-img"
              />
            ) : (
              <div className="admin-empty" style={{ padding: '2rem' }}>Chart not available</div>
            )}
          </div>
        ))}
        {Object.keys(currentCharts).length === 0 && (
          <div className="admin-empty" style={{ gridColumn: '1 / -1' }}>
            No chart data available. Send some messages to the chatbot first.
          </div>
        )}
      </div>

      {/* ---- Top Queries Table ---- */}
      <h3 className="admin-section-title" style={{ marginTop: '2rem' }}>Top Queries</h3>
      {topQueries.length > 0 ? (
        <div className="admin-table-wrapper">
          <table className="admin-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Query</th>
                <th>Count</th>
                <th>Last Asked</th>
              </tr>
            </thead>
            <tbody>
              {topQueries.map((q, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td className="admin-td-text">{q.query}</td>
                  <td>{q.count}</td>
                  <td className="admin-td-text admin-card-value-small">{q.last_asked ? q.last_asked.slice(0, 10) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="admin-empty">No queries recorded yet.</div>
      )}

      {/* ---- Unanswered Questions Table ---- */}
      <h3 className="admin-section-title" style={{ marginTop: '2rem' }}>Unanswered Questions</h3>
      {unanswered && unanswered.total_queries > 0 && (
        <div className="admin-cards-row" style={{ marginBottom: '1rem' }}>
          <div className="admin-card">
            <div className="admin-card-label">Unanswered</div>
            <div className="admin-card-value">{unanswered.total_unanswered}</div>
          </div>
          <div className="admin-card">
            <div className="admin-card-label">Total Queries</div>
            <div className="admin-card-value">{unanswered.total_queries}</div>
          </div>
          <div className="admin-card">
            <div className="admin-card-label">Unanswered Rate</div>
            <div className="admin-card-value">
              {unanswered.total_queries > 0
                ? `${((unanswered.total_unanswered / unanswered.total_queries) * 100).toFixed(1)}%`
                : '0%'}
            </div>
          </div>
        </div>
      )}
      <p className="admin-info-note">
        These are questions the chatbot could not answer. Consider adding relevant FAQs or
        database entries to cover these topics, then re-index.
      </p>
      {unanswered?.queries?.length > 0 ? (
        <div className="admin-table-wrapper">
          <table className="admin-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Question</th>
                <th>Times Asked</th>
                <th>Language</th>
                <th>Best FAQ Score</th>
                <th>Best DB Score</th>
                <th>Last Asked</th>
              </tr>
            </thead>
            <tbody>
              {unanswered.queries.map((q, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td className="admin-td-text" dir="auto">{q.query}</td>
                  <td>{q.count}</td>
                  <td>{q.language === 'ar' ? 'Arabic' : 'English'}</td>
                  <td>{(q.faq_top_score * 100).toFixed(0)}%</td>
                  <td>{(q.db_top_score * 100).toFixed(0)}%</td>
                  <td className="admin-td-text admin-card-value-small">{q.last_asked ? q.last_asked.slice(0, 10) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="admin-empty">No unanswered questions recorded yet.</div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Conversations & Feedback Tab
// ---------------------------------------------------------------------------
function ConversationsTab() {
  const [conversations, setConversations] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState(null); // null, 'positive', 'negative', 'unreviewed'
  const [stats, setStats] = useState(null);
  const [toast, setToast] = useState(null);

  // Expanded row state
  const [expandedId, setExpandedId] = useState(null);

  // Feedback form state
  const [feedbackTarget, setFeedbackTarget] = useState(null); // conversation being reviewed
  const [feedbackRating, setFeedbackRating] = useState(null);
  const [correctedAnswer, setCorrectedAnswer] = useState('');
  const [feedbackComment, setFeedbackComment] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const LIMIT = 20;

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [convData, statsData] = await Promise.all([
        getConversations(offset, LIMIT, filter),
        getFeedbackStats(),
      ]);
      setConversations(convData.conversations || []);
      setTotal(convData.total || 0);
      setStats(statsData);
    } catch (err) {
      setToast({ message: `Failed to load conversations: ${err.message}`, type: 'error' });
    } finally {
      setLoading(false);
    }
  }, [offset, filter]);

  useEffect(() => { loadData(); }, [loadData]);

  useEffect(() => {
    setOffset(0);
    setExpandedId(null);
  }, [filter]);

  const openFeedbackForm = (conv, rating) => {
    setFeedbackTarget(conv);
    setFeedbackRating(rating);
    setCorrectedAnswer('');
    setFeedbackComment('');
  };

  const closeFeedbackForm = () => {
    setFeedbackTarget(null);
    setFeedbackRating(null);
    setCorrectedAnswer('');
    setFeedbackComment('');
  };

  const handleSubmitFeedback = async () => {
    if (!feedbackTarget) return;
    if (feedbackRating === -1 && !correctedAnswer.trim() && !feedbackComment.trim()) {
      setToast({ message: 'Please provide a corrected answer or comment for negative feedback.', type: 'error' });
      return;
    }

    setSubmitting(true);
    try {
      await submitFeedback(
        feedbackTarget.id,
        feedbackRating,
        correctedAnswer.trim() || null,
        feedbackComment.trim() || null,
      );
      setToast({
        message: feedbackRating === 1 ? 'Marked as good answer.' : 'Feedback submitted. The corrected answer will be used for similar future queries.',
        type: 'success',
      });
      closeFeedbackForm();
      await loadData();
    } catch (err) {
      setToast({ message: `Failed to submit feedback: ${err.message}`, type: 'error' });
    } finally {
      setSubmitting(false);
    }
  };

  const handleQuickThumbsUp = async (conv) => {
    try {
      await submitFeedback(conv.id, 1, null, null);
      setToast({ message: 'Marked as good answer.', type: 'success' });
      await loadData();
    } catch (err) {
      setToast({ message: `Failed: ${err.message}`, type: 'error' });
    }
  };

  const totalPages = Math.ceil(total / LIMIT);
  const currentPage = Math.floor(offset / LIMIT) + 1;

  const formatDate = (ts) => {
    if (!ts) return '';
    const d = new Date(ts);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getRatingBadge = (conv) => {
    if (conv.rating === 1) return <span className="conv-badge conv-badge-positive">Good</span>;
    if (conv.rating === -1) return <span className="conv-badge conv-badge-negative">Needs Fix</span>;
    return <span className="conv-badge conv-badge-unreviewed">Unreviewed</span>;
  };

  return (
    <div className="admin-tab-content">
      {toast && <Toast {...toast} onClose={() => setToast(null)} />}

      {/* Stats Cards */}
      {stats && (
        <div className="admin-cards-row">
          <div className="admin-card">
            <div className="admin-card-label">Total Conversations</div>
            <div className="admin-card-value">{stats.total_conversations}</div>
          </div>
          <div className="admin-card">
            <div className="admin-card-label">Reviewed</div>
            <div className="admin-card-value">{stats.total_reviewed}</div>
          </div>
          <div className="admin-card">
            <div className="admin-card-label">Positive</div>
            <div className="admin-card-value" style={{ color: '#2e7d32' }}>{stats.positive}</div>
          </div>
          <div className="admin-card">
            <div className="admin-card-label">Negative</div>
            <div className="admin-card-value" style={{ color: '#c62828' }}>{stats.negative}</div>
          </div>
          <div className="admin-card">
            <div className="admin-card-label">Unreviewed</div>
            <div className="admin-card-value">{stats.unreviewed}</div>
          </div>
          <div className="admin-card">
            <div className="admin-card-label">With Corrections</div>
            <div className="admin-card-value">{stats.with_corrections}</div>
          </div>
        </div>
      )}

      <p className="admin-info-note">
        Review chatbot conversations and provide feedback. Negative feedback with a corrected answer
        will be used to improve responses for similar future questions.
      </p>

      {/* Filter Tabs */}
      <div className="admin-collection-tabs">
        {[
          { key: null, label: 'All' },
          { key: 'unreviewed', label: 'Unreviewed' },
          { key: 'positive', label: 'Positive' },
          { key: 'negative', label: 'Negative' },
        ].map((f) => (
          <button
            key={f.key || 'all'}
            className={`admin-collection-tab ${filter === f.key ? 'active' : ''}`}
            onClick={() => setFilter(f.key)}
          >
            {f.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="admin-loading">Loading conversations...</div>
      ) : (
        <>
          <div className="admin-table-info">
            Showing {conversations.length} of {total} conversations (page {currentPage} of {totalPages || 1})
          </div>

          <div className="admin-table-wrapper">
            <table className="admin-table">
              <thead>
                <tr>
                  <th style={{ width: '50px' }}>#</th>
                  <th>Question</th>
                  <th>Answer</th>
                  <th>Source</th>
                  <th>Lang</th>
                  <th>Status</th>
                  <th>Date</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {conversations.map((conv) => (
                  <React.Fragment key={conv.id}>
                    <tr
                      className={`conv-row ${expandedId === conv.id ? 'conv-row-expanded' : ''}`}
                      onClick={() => setExpandedId(expandedId === conv.id ? null : conv.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td className="admin-td-id">{conv.id}</td>
                      <td>
                        <span className="admin-td-text" dir="auto">{conv.query}</span>
                      </td>
                      <td>
                        <span className="admin-td-text admin-td-truncate" dir="auto">
                          {conv.answer ? conv.answer.substring(0, 120) + (conv.answer.length > 120 ? '...' : '') : ''}
                        </span>
                      </td>
                      <td className="conv-source-cell">
                        <span className="conv-source-tag">{conv.chosen_source || 'N/A'}</span>
                      </td>
                      <td>{conv.language === 'ar' ? 'AR' : 'EN'}</td>
                      <td>{getRatingBadge(conv)}</td>
                      <td className="admin-card-value-small">{formatDate(conv.created_at)}</td>
                      <td className="admin-td-actions" onClick={(e) => e.stopPropagation()}>
                        <button
                          className={`conv-thumb-btn conv-thumb-up ${conv.rating === 1 ? 'conv-thumb-active' : ''}`}
                          title="Good answer"
                          onClick={() => handleQuickThumbsUp(conv)}
                        >
                          &#x1F44D;
                        </button>
                        <button
                          className={`conv-thumb-btn conv-thumb-down ${conv.rating === -1 ? 'conv-thumb-active' : ''}`}
                          title="Needs correction"
                          onClick={() => openFeedbackForm(conv, -1)}
                        >
                          &#x1F44E;
                        </button>
                      </td>
                    </tr>

                    {/* Expanded Row */}
                    {expandedId === conv.id && (
                      <tr className="conv-expanded-row">
                        <td colSpan={8}>
                          <div className="conv-expanded-content">
                            <div className="conv-detail-section">
                              <h4>Question</h4>
                              <p dir="auto">{conv.query}</p>
                            </div>
                            <div className="conv-detail-section">
                              <h4>Full Answer</h4>
                              <p dir="auto" className="conv-full-answer">{conv.answer}</p>
                            </div>
                            <div className="conv-detail-meta">
                              <span>Source: <strong>{conv.chosen_source || 'N/A'}</strong></span>
                              <span>FAQ Score: <strong>{(conv.faq_top_score * 100).toFixed(0)}%</strong></span>
                              <span>DB Score: <strong>{(conv.db_top_score * 100).toFixed(0)}%</strong></span>
                              <span>Library Score: <strong>{(conv.library_top_score * 100).toFixed(0)}%</strong></span>
                              <span>Response Time: <strong>{conv.response_time_ms?.toFixed(0)}ms</strong></span>
                            </div>
                            {conv.corrected_answer && (
                              <div className="conv-detail-section conv-correction-section">
                                <h4>Admin Corrected Answer</h4>
                                <p dir="auto">{conv.corrected_answer}</p>
                                {conv.comment && <p className="conv-comment"><em>Comment: {conv.comment}</em></p>}
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
                {conversations.length === 0 && (
                  <tr><td colSpan={8} className="admin-empty">No conversations found.</td></tr>
                )}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="admin-pagination">
              <button
                className="admin-btn admin-btn-sm admin-btn-secondary"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - LIMIT))}
              >
                Previous
              </button>
              <span className="admin-page-info">Page {currentPage} of {totalPages}</span>
              <button
                className="admin-btn admin-btn-sm admin-btn-secondary"
                disabled={offset + LIMIT >= total}
                onClick={() => setOffset(offset + LIMIT)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}

      {/* Feedback Modal */}
      {feedbackTarget && (
        <div className="conv-modal-overlay" onClick={closeFeedbackForm}>
          <div className="conv-modal" onClick={(e) => e.stopPropagation()}>
            <h3 className="conv-modal-title">
              {feedbackRating === -1 ? 'Provide Correction' : 'Confirm Positive Feedback'}
            </h3>

            <div className="conv-modal-section">
              <label>Original Question:</label>
              <p dir="auto" className="conv-modal-text">{feedbackTarget.query}</p>
            </div>

            <div className="conv-modal-section">
              <label>Chatbot Answer:</label>
              <p dir="auto" className="conv-modal-text">{feedbackTarget.answer?.substring(0, 500)}</p>
            </div>

            {feedbackRating === -1 && (
              <>
                <div className="admin-form-group">
                  <label>What should the correct answer be?</label>
                  <textarea
                    value={correctedAnswer}
                    onChange={(e) => setCorrectedAnswer(e.target.value)}
                    rows={5}
                    placeholder="Type the correct answer that the chatbot should give for this question..."
                  />
                </div>
                <div className="admin-form-group">
                  <label>Additional comment (optional)</label>
                  <textarea
                    value={feedbackComment}
                    onChange={(e) => setFeedbackComment(e.target.value)}
                    rows={2}
                    placeholder="Any notes about why this answer was wrong..."
                  />
                </div>
              </>
            )}

            <div className="conv-modal-actions">
              <button
                className="admin-btn admin-btn-primary"
                onClick={handleSubmitFeedback}
                disabled={submitting}
              >
                {submitting ? 'Submitting...' : 'Submit Feedback'}
              </button>
              <button className="admin-btn admin-btn-secondary" onClick={closeFeedbackForm}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Admin Dashboard
// ---------------------------------------------------------------------------
function AdminDashboard() {
  const [activeTab, setActiveTab] = useState('status');

  return (
    <div className="app-container admin-page" dir="ltr" lang="en">
      <Header />

      <div className="admin-dashboard">
        <div className="admin-header">
          <h2 className="admin-title">Admin Dashboard</h2>
        </div>

        <div className="admin-tabs">
          <button
            className={`admin-tab ${activeTab === 'status' ? 'active' : ''}`}
            onClick={() => setActiveTab('status')}
          >
            System Status
          </button>
          <button
            className={`admin-tab ${activeTab === 'data' ? 'active' : ''}`}
            onClick={() => setActiveTab('data')}
          >
            Data Management
          </button>
          <button
            className={`admin-tab ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            Analytics
          </button>
          <button
            className={`admin-tab ${activeTab === 'conversations' ? 'active' : ''}`}
            onClick={() => setActiveTab('conversations')}
          >
            Conversations
          </button>
        </div>

        {activeTab === 'status' && <SystemStatusTab />}
        {activeTab === 'data' && <DataManagementTab />}
        {activeTab === 'analytics' && <AnalyticsTab />}
        {activeTab === 'conversations' && <ConversationsTab />}
      </div>

      <Footer />
    </div>
  );
}

export default AdminDashboard;
