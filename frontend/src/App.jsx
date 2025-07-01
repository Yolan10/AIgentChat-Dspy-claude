import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import Dashboard from './components/Dashboard';
import ConfigPanel from './components/ConfigPanel';
import PromptEditor from './components/PromptEditor';
import EffectivenessChart from './components/EffectivenessChart';
import ConversationViewer from './components/ConversationViewer';
import RunHistory from './components/RunHistory';
import TokenUsage from './components/TokenUsage';
import Login from './Login';
import { Settings, BarChart3, MessageSquare, History, FileText, Activity, Coins } from 'lucide-react';

// Configure socket connection for production
const socketURL = window.location.hostname === 'localhost' 
  ? 'http://localhost:5000' 
  : window.location.origin;

const socket = io(socketURL, {
  transports: ['websocket', 'polling'],
  upgrade: true,
  reconnection: true,
  reconnectionAttempts: 5,
  reconnectionDelay: 1000,
});

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [authenticated, setAuthenticated] = useState(false);
  const [simulationState, setSimulationState] = useState({
    running: false,
    paused: false,
    current_run: null,
    progress: { completed: 0, total: 0 }
  });
  const [conversationTurns, setConversationTurns] = useState([]);
  const [systemEvents, setSystemEvents] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Check authentication
    fetch('/api/check_auth')
      .then(res => res.json())
      .then(data => setAuthenticated(data.authenticated))
      .catch(err => {
        console.error('Auth check failed:', err);
        setAuthenticated(false);
        setError('Failed to check authentication status');
      });

    // Socket connection handlers
    socket.on('connect', () => {
      console.log('Connected to server');
      setError(null);
    });

    socket.on('connect_error', (err) => {
      console.error('Socket connection error:', err);
      setError('Connection to server lost. Please refresh the page.');
    });

    socket.on('status_update', (status) => {
      setSimulationState(status);
    });

    socket.on('conversation_turn', (turn) => {
      setConversationTurns(prev => [...prev.slice(-50), turn]);
    });

    socket.on('progress_update', (progress) => {
      setSimulationState(prev => ({ ...prev, progress }));
    });

    socket.on('system_event', (event) => {
      setSystemEvents(prev => [...prev.slice(-20), event]);
    });

    // Fetch initial status
    fetch('/api/status')
      .then(res => res.json())
      .then(setSimulationState)
      .catch(err => {
        console.error('Failed to fetch status:', err);
        setError('Failed to fetch simulation status');
      });

    return () => {
      socket.off('connect');
      socket.off('connect_error');
      socket.off('status_update');
      socket.off('conversation_turn');
      socket.off('progress_update');
      socket.off('system_event');
    };
  }, []);

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'config', label: 'Configuration', icon: Settings },
    { id: 'prompts', label: 'Prompt Editor', icon: FileText },
    { id: 'effectiveness', label: 'Effectiveness', icon: BarChart3 },
    { id: 'tokens', label: 'Token Usage', icon: Coins },
    { id: 'conversations', label: 'Conversations', icon: MessageSquare },
    { id: 'history', label: 'Run History', icon: History }
  ];

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <Dashboard
            simulationState={simulationState}
            conversationTurns={conversationTurns}
            systemEvents={systemEvents}
          />
        );
      case 'config':
        return <ConfigPanel />;
      case 'prompts':
        return <PromptEditor />;
      case 'effectiveness':
        return <EffectivenessChart />;
      case 'tokens':
        return <TokenUsage />;
      case 'conversations':
        return <ConversationViewer conversationTurns={conversationTurns} />;
      case 'history':
        return <RunHistory />;
      default:
        return <Dashboard simulationState={simulationState} />;
    }
  };

  const handleLogout = async () => {
    try {
      await fetch('/api/logout', { method: 'POST' });
      setAuthenticated(false);
    } catch (err) {
      console.error('Logout failed:', err);
      setError('Failed to logout');
    }
  };

  if (!authenticated) {
    return <Login onLogin={() => setAuthenticated(true)} />;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Error Banner */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 relative" role="alert">
          <span className="block sm:inline">{error}</span>
          <span className="absolute top-0 bottom-0 right-0 px-4 py-3">
            <button
              onClick={() => setError(null)}
              className="text-red-700 hover:text-red-900"
            >
              ×
            </button>
          </span>
        </div>
      )}

      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900">AI Agent Monitor</h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <span className={`inline-block w-3 h-3 rounded-full ${
                  simulationState.running 
                    ? simulationState.paused 
                      ? 'bg-yellow-500' 
                      : 'bg-green-500'
                    : 'bg-gray-400'
                }`}></span>
                <span className="text-sm font-medium text-gray-700">
                  {simulationState.running 
                    ? simulationState.paused 
                      ? 'Paused' 
                      : 'Running'
                    : 'Stopped'
                  }
                </span>
              </div>
              
              {simulationState.current_run && (
                <div className="text-sm text-gray-600">
                  Run #{simulationState.current_run}
                </div>
              )}
              <button 
                onClick={handleLogout} 
                className="px-4 py-2 bg-gray-200 text-gray-900 rounded-md hover:bg-gray-300 transition-colors text-sm"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderActiveTab()}
      </main>
    </div>
  );
}

export default App;
