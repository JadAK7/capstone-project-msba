const API_BASE = process.env.REACT_APP_API_URL || '';

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

const TOKEN_KEY = 'aub_admin_token';

export function getStoredToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setStoredToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearStoredToken() {
  localStorage.removeItem(TOKEN_KEY);
}

function adminHeaders(extra = {}) {
  const token = getStoredToken();
  const headers = { 'Content-Type': 'application/json', ...extra };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

function adminHeadersNoBody(extra = {}) {
  const token = getStoredToken();
  const headers = { ...extra };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

async function adminFetch(url, options = {}) {
  const res = await fetch(url, options);
  if (res.status === 401) {
    clearStoredToken();
    window.dispatchEvent(new Event('admin-logout'));
  }
  return res;
}

export async function adminLogin(username, password) {
  const res = await fetch(`${API_BASE}/api/admin/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(res.status === 401 ? 'Invalid username or password' : `Login failed: ${errorText}`);
  }
  const data = await res.json();
  setStoredToken(data.token);
  return data;
}

export async function verifyAdminToken() {
  const token = getStoredToken();
  if (!token) return false;
  try {
    const res = await fetch(`${API_BASE}/api/admin/verify-token`, {
      headers: { 'Authorization': `Bearer ${token}` },
    });
    return res.ok;
  } catch {
    return false;
  }
}

export function adminLogout() {
  clearStoredToken();
}

// ---------------------------------------------------------------------------
// Existing chatbot API
// ---------------------------------------------------------------------------

export async function sendMessage(message, history = []) {
  // Language is always auto-detected by the backend from the message text.
  // No language parameter is sent -- this ensures per-message detection works
  // correctly even when the user alternates between Arabic and English.
  const body = { message };
  if (history.length > 0) {
    body.history = history;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000);

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.status === 503) {
      throw new Error('SERVICE_UNAVAILABLE');
    }
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Server error: ${res.status} - ${errorText}`);
    }
    return res.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('TIMEOUT');
    }
    throw err;
  }
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Public feedback (no auth required — used from chat UI)
// ---------------------------------------------------------------------------

export async function submitPublicFeedback(conversationId, rating, comment = null) {
  const body = { conversation_id: conversationId, rating, source: 'user' };
  if (comment) body.comment = comment;

  const res = await fetch(`${API_BASE}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to submit feedback: ${res.status} - ${errorText}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Collections
// ---------------------------------------------------------------------------

export async function getCollections() {
  const res = await adminFetch(`${API_BASE}/api/admin/collections`, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch collections: ${res.status}`);
  return res.json();
}

export async function getCollectionEntries(name, offset = 0, limit = 20) {
  const res = await adminFetch(
    `${API_BASE}/api/admin/collections/${name}/entries?offset=${offset}&limit=${limit}`,
    { headers: adminHeadersNoBody() },
  );
  if (!res.ok) throw new Error(`Failed to fetch entries: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- FAQ CRUD
// ---------------------------------------------------------------------------

export async function addFAQ(question, answer) {
  const res = await adminFetch(`${API_BASE}/api/admin/faq`, {
    method: 'POST', headers: adminHeaders(), body: JSON.stringify({ question, answer }),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to add FAQ: ${res.status} - ${e}`); }
  return res.json();
}

export async function updateFAQ(id, question, answer) {
  const res = await adminFetch(`${API_BASE}/api/admin/faq/${id}`, {
    method: 'PUT', headers: adminHeaders(), body: JSON.stringify({ question, answer }),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to update FAQ: ${res.status} - ${e}`); }
  return res.json();
}

export async function deleteFAQ(id) {
  const res = await adminFetch(`${API_BASE}/api/admin/faq/${id}`, {
    method: 'DELETE', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to delete FAQ: ${res.status} - ${e}`); }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Database CRUD
// ---------------------------------------------------------------------------

export async function addDatabase(name, description) {
  const res = await adminFetch(`${API_BASE}/api/admin/database`, {
    method: 'POST', headers: adminHeaders(), body: JSON.stringify({ name, description }),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to add database: ${res.status} - ${e}`); }
  return res.json();
}

export async function updateDatabase(id, name, description) {
  const res = await adminFetch(`${API_BASE}/api/admin/database/${id}`, {
    method: 'PUT', headers: adminHeaders(), body: JSON.stringify({ name, description }),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to update database: ${res.status} - ${e}`); }
  return res.json();
}

export async function deleteDatabase(id) {
  const res = await adminFetch(`${API_BASE}/api/admin/database/${id}`, {
    method: 'DELETE', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to delete database: ${res.status} - ${e}`); }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Custom Notes CRUD
// ---------------------------------------------------------------------------

export async function addCustomNote(label, content) {
  const res = await adminFetch(`${API_BASE}/api/admin/custom-note`, {
    method: 'POST', headers: adminHeaders(), body: JSON.stringify({ label, content }),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to add custom note: ${res.status} - ${e}`); }
  return res.json();
}

export async function updateCustomNote(id, label, content) {
  const res = await adminFetch(`${API_BASE}/api/admin/custom-note/${id}`, {
    method: 'PUT', headers: adminHeaders(), body: JSON.stringify({ label, content }),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to update custom note: ${res.status} - ${e}`); }
  return res.json();
}

export async function deleteCustomNote(id) {
  const res = await adminFetch(`${API_BASE}/api/admin/custom-note/${id}`, {
    method: 'DELETE', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to delete custom note: ${res.status} - ${e}`); }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Library Pages
// ---------------------------------------------------------------------------

export async function searchDocumentChunks(query, offset = 0, limit = 20) {
  const res = await adminFetch(
    `${API_BASE}/api/admin/document-chunks/search?q=${encodeURIComponent(query)}&offset=${offset}&limit=${limit}`,
    { headers: adminHeadersNoBody() },
  );
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

export async function deleteDocumentChunk(id) {
  const res = await adminFetch(`${API_BASE}/api/admin/document-chunk/${id}`, {
    method: 'DELETE', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to delete chunk: ${res.status} - ${e}`); }
  return res.json();
}

export async function deleteLibraryPage(id) {
  const res = await adminFetch(`${API_BASE}/api/admin/library-page/${id}`, {
    method: 'DELETE', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to delete library page: ${res.status} - ${e}`); }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Rescrape library website
// ---------------------------------------------------------------------------

export async function triggerRescrape() {
  const res = await adminFetch(`${API_BASE}/api/admin/rescrape`, {
    method: 'POST', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Rescrape failed: ${res.status} - ${e}`); }
  return res.json();
}

export async function getRescrapeStatus() {
  const res = await adminFetch(`${API_BASE}/api/admin/rescrape/status`, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch scrape status: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Re-indexing & System Info
// ---------------------------------------------------------------------------

export async function triggerReindex() {
  const res = await adminFetch(`${API_BASE}/api/admin/reindex`, {
    method: 'POST', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Re-index failed: ${res.status} - ${e}`); }
  return res.json();
}

export async function getSystemInfo() {
  const res = await adminFetch(`${API_BASE}/api/admin/system-info`, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch system info: ${res.status}`);
  return res.json();
}

export async function clearCache() {
  const res = await adminFetch(`${API_BASE}/api/admin/clear-cache`, {
    method: 'POST', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to clear cache: ${res.status} - ${e}`); }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Analytics
// ---------------------------------------------------------------------------

export async function getAnalyticsSummary() {
  const res = await adminFetch(`${API_BASE}/api/admin/analytics/summary`, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch analytics summary: ${res.status}`);
  return res.json();
}

export async function getAnalyticsTrends() {
  const res = await adminFetch(`${API_BASE}/api/admin/analytics/trends`, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch analytics trends: ${res.status}`);
  return res.json();
}

export async function getTopQueries() {
  const res = await adminFetch(`${API_BASE}/api/admin/analytics/top-queries`, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch top queries: ${res.status}`);
  return res.json();
}

export async function getUnansweredQueries() {
  const res = await adminFetch(`${API_BASE}/api/admin/analytics/unanswered-queries`, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch unanswered queries: ${res.status}`);
  return res.json();
}

export async function getAnalyticsCharts() {
  const res = await adminFetch(`${API_BASE}/api/admin/analytics/charts`, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch analytics charts: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Conversations & Feedback
// ---------------------------------------------------------------------------

export async function getConversations(offset = 0, limit = 30, ratingFilter = null) {
  let url = `${API_BASE}/api/admin/conversations?offset=${offset}&limit=${limit}`;
  if (ratingFilter) url += `&rating_filter=${ratingFilter}`;
  const res = await adminFetch(url, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch conversations: ${res.status}`);
  return res.json();
}

export async function getConversation(id) {
  const res = await adminFetch(`${API_BASE}/api/admin/conversations/${id}`, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch conversation: ${res.status}`);
  return res.json();
}

export async function deleteConversation(id) {
  const res = await adminFetch(`${API_BASE}/api/admin/conversations/${id}`, {
    method: 'DELETE', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to delete conversation: ${res.status} - ${e}`); }
  return res.json();
}

export async function submitFeedback(conversationId, rating, correctedAnswer = null, comment = null, source = 'admin') {
  const body = { conversation_id: conversationId, rating, source };
  if (correctedAnswer) body.corrected_answer = correctedAnswer;
  if (comment) body.comment = comment;

  const res = await adminFetch(`${API_BASE}/api/admin/feedback`, {
    method: 'POST', headers: adminHeaders(), body: JSON.stringify(body),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to submit feedback: ${res.status} - ${e}`); }
  return res.json();
}

export async function deleteFeedback(feedbackId) {
  const res = await adminFetch(`${API_BASE}/api/admin/feedback/${feedbackId}`, {
    method: 'DELETE', headers: adminHeadersNoBody(),
  });
  if (!res.ok) throw new Error(`Failed to delete feedback: ${res.status}`);
  return res.json();
}

export async function deleteAllFeedback() {
  const res = await adminFetch(`${API_BASE}/api/admin/feedback/all`, {
    method: 'DELETE', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to delete all feedback: ${res.status} - ${e}`); }
  return res.json();
}

export async function getFeedbackStats() {
  const res = await adminFetch(`${API_BASE}/api/admin/feedback/stats`, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch feedback stats: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Evaluation
// ---------------------------------------------------------------------------

export async function runEvaluation(questions, language = null) {
  const body = { questions };
  if (language) body.language = language;

  const res = await adminFetch(`${API_BASE}/api/admin/evaluation/run`, {
    method: 'POST', headers: adminHeaders(), body: JSON.stringify(body),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Evaluation failed: ${res.status} - ${e}`); }
  return res.json();
}

// ---------------------------------------------------------------------------
// Escalation API
// ---------------------------------------------------------------------------

export async function submitEscalation(studentEmail, studentName, question) {
  const res = await fetch(`${API_BASE}/api/escalate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_email: studentEmail, student_name: studentName, question }),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to submit escalation: ${res.status} - ${errorText}`);
  }
  return res.json();
}

export async function getEscalations(statusFilter = null, offset = 0, limit = 30) {
  let url = `${API_BASE}/api/admin/escalations?offset=${offset}&limit=${limit}`;
  if (statusFilter) url += `&status_filter=${statusFilter}`;
  const res = await adminFetch(url, { headers: adminHeadersNoBody() });
  if (!res.ok) throw new Error(`Failed to fetch escalations: ${res.status}`);
  return res.json();
}


export async function deleteEscalation(escalationId) {
  const res = await adminFetch(`${API_BASE}/api/admin/escalations/${escalationId}`, {
    method: 'DELETE', headers: adminHeadersNoBody(),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Failed to delete escalation: ${res.status} - ${e}`); }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Evaluation
// ---------------------------------------------------------------------------

export async function runSingleEvaluation(question, language = null) {
  const body = { question };
  if (language) body.language = language;

  const res = await adminFetch(`${API_BASE}/api/admin/evaluation/single`, {
    method: 'POST', headers: adminHeaders(), body: JSON.stringify(body),
  });
  if (!res.ok) { const e = await res.text(); throw new Error(`Evaluation failed: ${res.status} - ${e}`); }
  return res.json();
}
