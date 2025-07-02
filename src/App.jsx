import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { config } from './config';
import Dashboard from './components/Dashboard';
import ConfigPanel from './components/ConfigPanel';
import PromptEditor from './components/PromptEditor';
import EffectivenessChart from './components/EffectivenessChart';
import ConversationViewer from './components/ConversationViewer';
import RunHistory from './components/RunHistory';
import TokenUsage from './components/TokenUsage';
import Login from './Login';
import { Settings, BarChart3, MessageSquare, History, FileText, Activity, Coins } from 'lucide-react';

const socket = io(config.WS_URL, {
  withCredentials: true
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

  useEffect(() => {
    fetch(config.API_ENDPOINTS.checkAuth, {
      credentials: 'include'
    })
      .then(res => res.json())
      .then(data => setAuthenticated(data.authenticated))
      .catch(() => setAuthenticated(false));

    // Socket event listeners
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
    fetch(config.API_ENDPOINTS.status, {
      credentials: 'include'
    })
      .then(res => res.json())
      .then(setSimulationState)
      .catch(console.error);

    return () => {
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
    await fetch(config.API_ENDPOINTS.logout, { 
      method: 'POST', 
      credentials: 'include' 
    });
    setAuthenticated(false);
  };

  if (!authenticated) {
    return <Login onLogin={() => setAuthenticated(true)} />;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900">AI Agent Monitor</h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <span className={`status-indicator ${
                  simulationState.running 
                    ? simulationState.paused 
                      ? 'status-paused' 
                      : 'status-running'
                    : 'status-stopped'
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
              <button onClick={handleLogout} className="btn-secondary text-sm">Logout</button>
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
                      ? 'border-primary-500 text-primary-600'
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
