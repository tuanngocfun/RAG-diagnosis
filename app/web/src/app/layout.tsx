import '../styles/globals.css';

export const metadata = {
  title: 'Medical RAG Chatbot',
  description: 'AI-powered chatbot for medical case analysis using multimodal RAG',
};

interface RootLayoutProps {
  children: any;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}