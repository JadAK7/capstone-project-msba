import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';

const LanguageContext = createContext();

export function LanguageProvider({ children }) {
  // UI display language (controls i18n strings for header, footer, buttons, etc.)
  // Initialize from localStorage if the user previously toggled.
  const [language, setLanguage] = useState(() => {
    return localStorage.getItem('aub-lang') || 'en';
  });

  // Persist language choice and update document-level attributes for RTL/LTR.
  // NOTE: This controls the overall page layout direction (for UI chrome).
  // Individual message bubbles handle their own dir based on message content.
  useEffect(() => {
    localStorage.setItem('aub-lang', language);
    document.documentElement.setAttribute('dir', language === 'ar' ? 'rtl' : 'ltr');
    document.documentElement.setAttribute('lang', language);
  }, [language]);

  const toggleLanguage = useCallback(() => {
    setLanguage((prev) => (prev === 'en' ? 'ar' : 'en'));
  }, []);

  const isRTL = language === 'ar';

  return (
    <LanguageContext.Provider value={{
      language,
      setLanguage,
      toggleLanguage,
      isRTL,
    }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}
