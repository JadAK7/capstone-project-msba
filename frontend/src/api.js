const API_BASE = process.env.REACT_APP_API_URL || '';

// ---------------------------------------------------------------------------
// Existing chatbot API
// ---------------------------------------------------------------------------

export async function sendMessage(message, language = null) {
  const body = { message };
  if (language) {
    body.language = language; // "en", "ar", or null for auto-detect
  }

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Server error: ${res.status} - ${errorText}`);
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Collections
// ---------------------------------------------------------------------------

export async function getCollections() {
  const res = await fetch(`${API_BASE}/api/admin/collections`);
  if (!res.ok) throw new Error(`Failed to fetch collections: ${res.status}`);
  return res.json();
}

export async function getCollectionEntries(name, offset = 0, limit = 20) {
  const res = await fetch(
    `${API_BASE}/api/admin/collections/${name}/entries?offset=${offset}&limit=${limit}`
  );
  if (!res.ok) throw new Error(`Failed to fetch entries: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- FAQ CRUD
// ---------------------------------------------------------------------------

export async function addFAQ(question, answer) {
  const res = await fetch(`${API_BASE}/api/admin/faq`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, answer }),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to add FAQ: ${res.status} - ${errorText}`);
  }
  return res.json();
}

export async function updateFAQ(id, question, answer) {
  const res = await fetch(`${API_BASE}/api/admin/faq/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, answer }),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to update FAQ: ${res.status} - ${errorText}`);
  }
  return res.json();
}

export async function deleteFAQ(id) {
  const res = await fetch(`${API_BASE}/api/admin/faq/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to delete FAQ: ${res.status} - ${errorText}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Database CRUD
// ---------------------------------------------------------------------------

export async function addDatabase(name, description) {
  const res = await fetch(`${API_BASE}/api/admin/database`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description }),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to add database: ${res.status} - ${errorText}`);
  }
  return res.json();
}

export async function updateDatabase(id, name, description) {
  const res = await fetch(`${API_BASE}/api/admin/database/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description }),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to update database: ${res.status} - ${errorText}`);
  }
  return res.json();
}

export async function deleteDatabase(id) {
  const res = await fetch(`${API_BASE}/api/admin/database/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to delete database: ${res.status} - ${errorText}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Library Pages
// ---------------------------------------------------------------------------

export async function deleteLibraryPage(id) {
  const res = await fetch(`${API_BASE}/api/admin/library-page/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to delete library page: ${res.status} - ${errorText}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Re-indexing & System Info
// ---------------------------------------------------------------------------

export async function triggerReindex() {
  const res = await fetch(`${API_BASE}/api/admin/reindex`, {
    method: 'POST',
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Re-index failed: ${res.status} - ${errorText}`);
  }
  return res.json();
}

export async function getSystemInfo() {
  const res = await fetch(`${API_BASE}/api/admin/system-info`);
  if (!res.ok) throw new Error(`Failed to fetch system info: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Analytics
// ---------------------------------------------------------------------------

export async function getAnalyticsSummary() {
  const res = await fetch(`${API_BASE}/api/admin/analytics/summary`);
  if (!res.ok) throw new Error(`Failed to fetch analytics summary: ${res.status}`);
  return res.json();
}

export async function getAnalyticsTrends() {
  const res = await fetch(`${API_BASE}/api/admin/analytics/trends`);
  if (!res.ok) throw new Error(`Failed to fetch analytics trends: ${res.status}`);
  return res.json();
}

export async function getTopQueries() {
  const res = await fetch(`${API_BASE}/api/admin/analytics/top-queries`);
  if (!res.ok) throw new Error(`Failed to fetch top queries: ${res.status}`);
  return res.json();
}
