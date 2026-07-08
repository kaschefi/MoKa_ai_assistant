import React from 'react';
import MokaLanding from './components/MokaLanding';
import ChatInterface from './components/ChatInterface';
import { useLocation } from './hooks/useLocation';

/**
 * Main App component.
 * Manages routing dynamically based on URL pathnames.
 */
export const App: React.FC = () => {
  const [path, navigate] = useLocation();

  return path === '/chat' || path === '/chat/' ? (
    <ChatInterface onBackToLanding={() => navigate('/')} />
  ) : (
    <MokaLanding onStartChat={() => navigate('/chat')} />
  );
};

export default App;

