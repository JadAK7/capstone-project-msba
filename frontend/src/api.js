const API_BASE = process.env.REACT_APP_API_URL || '';

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

export async function searchDocumentChunks(query, offset = 0, limit = 20) {
  const res = await fetch(
    `${API_BASE}/api/admin/document-chunks/search?q=${encodeURIComponent(query)}&offset=${offset}&limit=${limit}`
  );
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

export async function deleteDocumentChunk(id) {
  const res = await fetch(`${API_BASE}/api/admin/document-chunk/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to delete chunk: ${res.status} - ${errorText}`);
  }
  return res.json();
}

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
// Admin API -- Rescrape library website
// ---------------------------------------------------------------------------

export async function triggerRescrape() {
  const res = await fetch(`${API_BASE}/api/admin/rescrape`, {
    method: 'POST',
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Rescrape failed: ${res.status} - ${errorText}`);
  }
  return res.json();
}

export async function getRescrapeStatus() {
  const res = await fetch(`${API_BASE}/api/admin/rescrape/status`);
  if (!res.ok) throw new Error(`Failed to fetch scrape status: ${res.status}`);
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

export async function getUnansweredQueries() {
  const res = await fetch(`${API_BASE}/api/admin/analytics/unanswered-queries`);
  if (!res.ok) throw new Error(`Failed to fetch unanswered queries: ${res.status}`);
  return res.json();
}

export async function getAnalyticsCharts() {
  const res = await fetch(`${API_BASE}/api/admin/analytics/charts`);
  if (!res.ok) throw new Error(`Failed to fetch analytics charts: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Conversations & Feedback
// ---------------------------------------------------------------------------

export async function getConversations(offset = 0, limit = 30, ratingFilter = null) {
  let url = `${API_BASE}/api/admin/conversations?offset=${offset}&limit=${limit}`;
  if (ratingFilter) url += `&rating_filter=${ratingFilter}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch conversations: ${res.status}`);
  return res.json();
}

export async function getConversation(id) {
  const res = await fetch(`${API_BASE}/api/admin/conversations/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch conversation: ${res.status}`);
  return res.json();
}

export async function deleteConversation(id) {
  const res = await fetch(`${API_BASE}/api/admin/conversations/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to delete conversation: ${res.status} - ${errorText}`);
  }
  return res.json();
}

export async function submitFeedback(conversationId, rating, correctedAnswer = null, comment = null, source = 'admin') {
  const body = { conversation_id: conversationId, rating, source };
  if (correctedAnswer) body.corrected_answer = correctedAnswer;
  if (comment) body.comment = comment;

  const res = await fetch(`${API_BASE}/api/admin/feedback`, {
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

export async function deleteFeedback(feedbackId) {
  const res = await fetch(`${API_BASE}/api/admin/feedback/${feedbackId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete feedback: ${res.status}`);
  return res.json();
}

export async function getFeedbackStats() {
  const res = await fetch(`${API_BASE}/api/admin/feedback/stats`);
  if (!res.ok) throw new Error(`Failed to fetch feedback stats: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Admin API -- Evaluation
// ---------------------------------------------------------------------------

export async function runEvaluation(questions, language = null) {
  const body = { questions };
  if (language) body.language = language;

  const res = await fetch(`${API_BASE}/api/admin/evaluation/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Evaluation failed: ${res.status} - ${errorText}`);
  }
  return res.json();
}

export async function runSingleEvaluation(question, language = null) {
  const body = { question };
  if (language) body.language = language;

  const res = await fetch(`${API_BASE}/api/admin/evaluation/single`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Evaluation failed: ${res.status} - ${errorText}`);
  }
  return res.json();
}
