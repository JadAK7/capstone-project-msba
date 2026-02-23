import React from 'react';
import { useLanguage } from '../LanguageContext';
import { t } from '../i18n';

function Footer() {
  const { language } = useLanguage();

  return (
    <footer className="app-footer">
      <div className="footer-content">
        <p className="footer-title">{t(language, 'footerTitle')}</p>
        <div className="footer-divider"></div>
        <div className="footer-links">
          <a
            href="https://www.aub.edu.lb/library"
            target="_blank"
            rel="noopener noreferrer"
            className="footer-link"
          >
            {t(language, 'footerLinkLibraries')}
          </a>
          <span className="footer-separator">|</span>
          <a
            href="https://www.aub.edu.lb"
            target="_blank"
            rel="noopener noreferrer"
            className="footer-link"
          >
            {t(language, 'footerLinkHomepage')}
          </a>
          <span className="footer-separator">|</span>
          <a
            href="https://www.aub.edu.lb/library/Pages/contact-us.aspx"
            target="_blank"
            rel="noopener noreferrer"
            className="footer-link"
          >
            {t(language, 'footerLinkContact')}
          </a>
        </div>
        <p className="footer-powered">{t(language, 'footerPowered')}</p>
      </div>
    </footer>
  );
}

export default Footer;
