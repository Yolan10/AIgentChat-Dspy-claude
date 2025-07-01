import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import { SupabaseProvider } from './components/SupabaseProvider'
import { DemoDataProvider } from './components/DemoDataProvider'
import { useAuth } from './hooks/useSupabase'
import { ConnectionStatus } from './components/ConnectionStatus'
import { config } from './lib/config'

// Import components
import Dashboard from './components/Dashboard'
import ConfigPanel from './components/ConfigPanel'
import PromptEditor from './components/PromptEditor'
import EffectivenessChart from './components/EffectivenessChart'
import ConversationViewer from './components/ConversationViewer'
import RunHistory from './components/RunHistory'
import TokenUsage from './components/TokenUsage'
import Login from './components/Login'
import { Settings, BarChart3, MessageSquare, History, FileText, Activity, Coins } from 'lucide-react'

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      cacheTime: 1000 * 60 * 10, // 10 minutes
      retry: (failureCount, error) => {
        // Don't retry in demo mode
        if (config.app.isDevelopment && !config.supabase.url) {
          return false
        }
        return failureCount < 3
      },
    },
  },
})

function AppContent() {
  const { user, loading } = useAuth()
  const [activeTab, setActiveTab] = useState('dashboard')

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'config', label: 'Configuration', icon: Settings },
    { id: 'prompts', label: 'Prompt Editor', icon: FileText },
    { id: 'effectiveness', label: 'Effectiveness', icon: BarChart3 },
    { id: 'tokens', label: 'Token Usage', icon: Coins },
    { id: 'conversations', label: 'Conversations', icon: MessageSquare },
    { id: 'history', label: 'Run History', icon: History }
  ]

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />
      case 'config':
        return <ConfigPanel />
      case 'prompts':
        return <PromptEditor />
      case 'effectiveness':
        return <EffectivenessChart />
      case 'tokens':
        return <TokenUsage />
      case 'conversations':
        return <ConversationViewer />
      case 'history':
        return <RunHistory />
      default:
        return <Dashboard />
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  // In demo mode, skip authentication
  const isDemoMode = !config.supabase.url || !config.supabase.anonKey
  if (!isDemoMode && !user) {
    return <Login />
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Environment Banner */}
      {config.app.isDevelopment && (
        <div className="bg-amber-500 text-white px-4 py-2 text-center text-sm font-medium">
          🚧 {isDemoMode ? 'DEMO MODE' : 'DEVELOPMENT MODE'} - {isDemoMode ? 'Supabase not configured' : 'Development environment with Supabase integration'}
        </div>
      )}

      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900">AI Agent Monitor</h1>
              {isDemoMode && (
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  Demo
                </span>
              )}
            </div>
            
            <div className="flex items-center space-x-4">
              {!isDemoMode && <ConnectionStatus />}
              
              <div className="text-sm text-gray-600">
                {isDemoMode ? 'Demo User' : user?.email}
              </div>
              
              <button 
                onClick={() => window.location.reload()} 
                className="btn-secondary text-sm"
              >
                Refresh
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
              const Icon = tab.icon
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
              )
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderActiveTab()}
      </main>

      <Toaster position="top-right" />
    </div>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SupabaseProvider>
        <DemoDataProvider>
          <Router>
            <Routes>
              <Route path="/*" element={<AppContent />} />
            </Routes>
          </Router>
        </DemoDataProvider>
      </SupabaseProvider>
    </QueryClientProvider>
  )
}

export default App