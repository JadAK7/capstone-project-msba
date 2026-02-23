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
