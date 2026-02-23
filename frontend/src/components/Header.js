import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';

function Header() {
  const { language, toggleLanguage } = useLanguage();
  const location = useLocation();
  const isAdmin = location.pathname === '/admin';

  return (
    <header className="app-header">
      <div className="header-content">
        <div className="header-top-row">
          <div className="header-nav-left">
            {isAdmin ? (
              <Link to="/" className="header-nav-link">Back to Chat</Link>
            ) : (
              <Link to="/admin" className="header-nav-link">Admin</Link>
            )}
          </div>
          <h1 className="header-title">{t(language, 'headerTitle')}</h1>
          <button
            onClick={toggleLanguage}
            className="language-toggle header-lang-btn"
            aria-label="Toggle language"
          >
            {t(language, 'langToggleLabel')}
          </button>
        </div>
        <p className="header-subtitle">{t(language, 'headerSubtitle')}</p>
      </div>
      <div className="header-accent-bar"></div>
    </header>
  );
}

export default Header;
