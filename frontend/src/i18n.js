const translations = {
  en: {
    headerTitle: 'AUB Libraries Assistant',
    headerSubtitle: 'American University of Beirut — University Libraries',
    subtitleBanner: {
      prefix: 'Get instant answers',
      suffix: ' to library questions and personalized database recommendations',
    },
    statusConnecting: 'Connecting to backend...',
    welcomeTitle: 'Welcome to AUB Libraries Assistant',
    welcomeDescription: 'Ask me about library services, resources, or get personalized database recommendations.',
    exampleQuestionsTitle: 'Example questions:',
    exampleQuestions: [
      'How do I access online databases?',
      'What are the library opening hours?',
      'I need databases for medical research',
      'How can I renew my library books?',
    ],
    roleUser: 'You',
    roleAssistant: 'Assistant',
    loadingText: 'Assistant is thinking...',
    inputPlaceholder: 'Ask about library services or database recommendations...',
    sendButton: 'Send',
    debugTitle: 'Debug Info',
    debugChosenSource: 'Chosen Source:',
    debugDbIntent: 'DB Keyword Intent:',
    debugLibraryAvailable: 'Library Pages Available:',
    debugDetectedLang: 'Detected Language:',
    yes: 'Yes',
    no: 'No',
    footerTitle: 'American University of Beirut',
    footerLinkLibraries: 'AUB Libraries',
    footerLinkHomepage: 'AUB Homepage',
    footerLinkContact: 'Contact Us',
    footerPowered: 'Powered by OpenAI',
    langToggleLabel: 'عربي',
    escalation: {
      buttonTitle: 'Ask a Librarian',
      modalTitle: 'Ask a Librarian',
      modalDescription: 'Can\'t find what you need? Send your question to a librarian and get a response by email.',
      nameLabel: 'Your Name',
      namePlaceholder: 'Enter your name',
      emailLabel: 'Your Email',
      emailPlaceholder: 'Enter your email address',
      questionLabel: 'Your Question',
      questionPlaceholder: 'Describe what you need help with...',
      submitButton: 'Send to Librarian',
      cancelButton: 'Cancel',
      successMessage: 'Your question has been sent to a librarian. You will receive a response at your email address.',
      errorMessage: 'Failed to send your question. Please try again.',
      sending: 'Sending...',
    },
  },
  ar: {
    headerTitle: 'مساعد مكتبات الجامعة الأمريكية',
    headerSubtitle: 'الجامعة الأمريكية في بيروت — مكتبات الجامعة',
    subtitleBanner: {
      prefix: 'احصل على إجابات فورية',
      suffix: ' لأسئلة المكتبة وتوصيات قواعد بيانات مخصصة',
    },
    statusConnecting: 'جارٍ الاتصال بالخادم...',
    welcomeTitle: 'مرحباً بك في مساعد مكتبات الجامعة',
    welcomeDescription: 'اسألني عن خدمات المكتبة أو الموارد أو احصل على توصيات مخصصة لقواعد البيانات.',
    exampleQuestionsTitle: 'أمثلة على الأسئلة:',
    exampleQuestions: [
      'كيف يمكنني الوصول إلى قواعد البيانات الإلكترونية؟',
      'ما هي ساعات عمل المكتبة؟',
      'أحتاج قواعد بيانات للبحث الطبي',
      'كيف يمكنني تجديد كتبي المستعارة؟',
    ],
    roleUser: 'أنت',
    roleAssistant: 'المساعد',
    loadingText: 'المساعد يفكر...',
    inputPlaceholder: 'اسأل عن خدمات المكتبة أو توصيات قواعد البيانات...',
    sendButton: 'إرسال',
    debugTitle: 'معلومات التصحيح',
    debugChosenSource: 'المصدر المختار:',
    debugDbIntent: 'كلمة مفتاحية لقاعدة بيانات:',
    debugLibraryAvailable: 'صفحات المكتبة متاحة:',
    debugDetectedLang: 'اللغة المكتشفة:',
    yes: 'نعم',
    no: 'لا',
    footerTitle: 'الجامعة الأمريكية في بيروت',
    footerLinkLibraries: 'مكتبات الجامعة',
    footerLinkHomepage: 'الصفحة الرئيسية',
    footerLinkContact: 'اتصل بنا',
    footerPowered: 'مدعوم بواسطة OpenAI',
    langToggleLabel: 'EN',
    escalation: {
      buttonTitle: 'اسأل أمين المكتبة',
      modalTitle: 'اسأل أمين المكتبة',
      modalDescription: 'لم تجد ما تبحث عنه؟ أرسل سؤالك إلى أمين المكتبة واحصل على رد عبر البريد الإلكتروني.',
      nameLabel: 'اسمك',
      namePlaceholder: 'أدخل اسمك',
      emailLabel: 'بريدك الإلكتروني',
      emailPlaceholder: 'أدخل بريدك الإلكتروني',
      questionLabel: 'سؤالك',
      questionPlaceholder: 'صف ما تحتاج المساعدة فيه...',
      submitButton: 'أرسل إلى أمين المكتبة',
      cancelButton: 'إلغاء',
      successMessage: 'تم إرسال سؤالك إلى أمين المكتبة. ستتلقى رداً على بريدك الإلكتروني.',
      errorMessage: 'فشل إرسال سؤالك. يرجى المحاولة مرة أخرى.',
      sending: 'جارٍ الإرسال...',
    },
  },
};

export function t(lang, key) {
  const keys = key.split('.');
  let value = translations[lang] || translations.en;
  for (const k of keys) {
    if (value && typeof value === 'object' && k in value) {
      value = value[k];
    } else {
      let fallback = translations.en;
      for (const fk of keys) {
        fallback = fallback?.[fk];
      }
      return fallback || key;
    }
  }
  return value;
}

export default translations;
