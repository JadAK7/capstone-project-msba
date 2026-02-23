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
  getSystemInfo,
  checkHealth,
  getAnalyticsSummary,
  getAnalyticsTrends,
  getTopQueries,
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
                  {activeCollection === 'library_pages' && <><th>Title</th><th>URL</th></>}
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
                  <tr><td colSpan={4} className="admin-empty">No entries found.</td></tr>
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
    </div>
  );
}

// ---------------------------------------------------------------------------
// Analytics Tab
// ---------------------------------------------------------------------------
function AnalyticsTab() {
  const [summary, setSummary] = useState(null);
  const [trends, setTrends] = useState([]);
  const [topQueries, setTopQueries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState(null);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const [s, t, q] = await Promise.all([
          getAnalyticsSummary(),
          getAnalyticsTrends(),
          getTopQueries(),
        ]);
        setSummary(s);
        setTrends(t);
        setTopQueries(q);
      } catch (err) {
        setToast({ message: `Failed to load analytics: ${err.message}`, type: 'error' });
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) return <div className="admin-loading">Loading analytics...</div>;

  const maxTrendCount = Math.max(1, ...trends.map((t) => t.count));

  return (
    <div className="admin-tab-content">
      {toast && <Toast {...toast} onClose={() => setToast(null)} />}

      <h3 className="admin-section-title">Conversation Summary</h3>
      <div className="admin-cards-row">
        <div className="admin-card">
          <div className="admin-card-label">Total Conversations</div>
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
      </div>

      <h3 className="admin-section-title">Language Distribution</h3>
      <div className="admin-bar-chart">
        {Object.entries(summary?.language_distribution || {}).map(([lang, pct]) => (
          <div className="admin-bar-row" key={lang}>
            <span className="admin-bar-label">{lang === 'en' ? 'English' : lang === 'ar' ? 'Arabic' : lang}</span>
            <div className="admin-bar-track">
              <div
                className="admin-bar-fill admin-bar-fill-primary"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="admin-bar-value">{pct}%</span>
          </div>
        ))}
        {Object.keys(summary?.language_distribution || {}).length === 0 && (
          <div className="admin-empty">No data yet.</div>
        )}
      </div>

      <h3 className="admin-section-title">Intent Distribution</h3>
      <div className="admin-bar-chart">
        {Object.entries(summary?.intent_distribution || {}).map(([intent, pct]) => (
          <div className="admin-bar-row" key={intent}>
            <span className="admin-bar-label">{intent}</span>
            <div className="admin-bar-track">
              <div
                className="admin-bar-fill admin-bar-fill-accent"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="admin-bar-value">{pct}%</span>
          </div>
        ))}
        {Object.keys(summary?.intent_distribution || {}).length === 0 && (
          <div className="admin-empty">No data yet.</div>
        )}
      </div>

      <h3 className="admin-section-title">Daily Trends (Last 30 Days)</h3>
      {trends.length > 0 ? (
        <div className="admin-trends-chart">
          {trends.map((day) => (
            <div className="admin-trend-bar-col" key={day.date} title={`${day.date}: ${day.count} conversations`}>
              <div className="admin-trend-bar-wrapper">
                <div
                  className="admin-trend-bar"
                  style={{ height: `${(day.count / maxTrendCount) * 100}%` }}
                />
              </div>
              <span className="admin-trend-label">
                {day.date.slice(5)}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className="admin-empty">No trend data yet.</div>
      )}

      <h3 className="admin-section-title">Top Queries</h3>
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
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Admin Dashboard
// ---------------------------------------------------------------------------
function AdminDashboard() {
  const [activeTab, setActiveTab] = useState('status');

  return (
    <div className="app-container" dir="ltr" lang="en">
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
        </div>

        {activeTab === 'status' && <SystemStatusTab />}
        {activeTab === 'data' && <DataManagementTab />}
        {activeTab === 'analytics' && <AnalyticsTab />}
      </div>

      <Footer />
    </div>
  );
}

export default AdminDashboard;
