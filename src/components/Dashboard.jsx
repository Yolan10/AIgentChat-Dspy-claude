import React, { useState, useEffect } from 'react'
import { Play, Pause, Square, RefreshCw, AlertCircle } from 'lucide-react'
import { useDemoData } from './DemoDataProvider'
import { config } from '@/lib/config'

const Dashboard = () => {
  const { runs, isLoading, isDemoMode } = useDemoData()
  const [formData, setFormData] = useState({
    instruction: 'Generate diverse hearing loss population',
    population_size: 36,
    goal: 'Uncover key insights about hearing loss experiences and generate a structured research plan'
  })

  const [simulationState, setSimulationState] = useState({
    running: false,
    paused: false,
    current_run: null,
    progress: { completed: 0, total: 0 }
  })

  const [conversationTurns, setConversationTurns] = useState([])
  const [systemEvents, setSystemEvents] = useState([])

  // Simulate real-time updates in demo mode
  useEffect(() => {
    if (isDemoMode && runs?.length > 0) {
      const runningRun = runs.find(run => run.status === 'running')
      if (runningRun) {
        setSimulationState({
          running: true,
          paused: false,
          current_run: runningRun.id,
          progress: { completed: Math.floor(runningRun.progress * runningRun.population_size / 100), total: runningRun.population_size }
        })

        // Simulate conversation turns
        const interval = setInterval(() => {
          const mockTurn = {
            speaker: Math.random() > 0.5 ? 'wizard' : 'pop',
            text: Math.random() > 0.5 
              ? 'Can you tell me more about your experience with hearing aids?'
              : 'I find it challenging in noisy environments like restaurants.',
            timestamp: new Date().toISOString(),
            agent_id: `${runningRun.id}.${Math.floor(Math.random() * 5) + 1}`
          }
          setConversationTurns(prev => [...prev.slice(-10), mockTurn])
        }, 3000)

        return () => clearInterval(interval)
      }
    }
  }, [isDemoMode, runs])

  const handleStart = async () => {
    if (isDemoMode) {
      // Demo mode simulation
      setSimulationState(prev => ({ ...prev, running: true, paused: false }))
      setSystemEvents(prev => [...prev, {
        type: 'simulation_start',
        timestamp: new Date().toISOString(),
        data: { message: 'Demo simulation started' }
      }])
    } else {
      // Real API call would go here
      console.log('Starting real simulation...')
    }
  }

  const handlePause = async () => {
    setSimulationState(prev => ({ 
      ...prev, 
      paused: !prev.paused 
    }))
  }

  const handleStop = async () => {
    setSimulationState({
      running: false,
      paused: false,
      current_run: null,
      progress: { completed: 0, total: 0 }
    })
  }

  const progressPercentage = simulationState.progress.total > 0 
    ? (simulationState.progress.completed / simulationState.progress.total) * 100 
    : 0

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        <span className="ml-3 text-gray-600">Loading dashboard...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Demo Mode Banner */}
      {isDemoMode && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertCircle className="w-5 h-5 text-blue-600 mr-2" />
            <div>
              <h3 className="text-sm font-medium text-blue-800">Demo Mode Active</h3>
              <p className="text-sm text-blue-700 mt-1">
                Supabase is not configured. Showing demo data with simulated functionality.
                {config.app.isDevelopment && ' Configure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to enable full functionality.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Control Panel */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Simulation Control</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="label">Instruction</label>
            <input
              type="text"
              className="input"
              value={formData.instruction}
              onChange={(e) => setFormData(prev => ({ ...prev, instruction: e.target.value }))}
              disabled={simulationState.running}
            />
          </div>
          
          <div>
            <label className="label">Population Size</label>
            <input
              type="number"
              className="input"
              value={formData.population_size}
              onChange={(e) => setFormData(prev => ({ ...prev, population_size: parseInt(e.target.value) }))}
              disabled={simulationState.running}
            />
          </div>
          
          <div>
            <label className="label">Wizard Goal</label>
            <input
              type="text"
              className="input"
              value={formData.goal}
              onChange={(e) => setFormData(prev => ({ ...prev, goal: e.target.value }))}
              disabled={simulationState.running}
            />
          </div>
        </div>
        
        <div className="flex space-x-3">
          <button
            onClick={handleStart}
            disabled={simulationState.running}
            className="btn-primary flex items-center space-x-2"
          >
            <Play className="w-4 h-4" />
            <span>Start</span>
          </button>
          
          <button
            onClick={handlePause}
            disabled={!simulationState.running}
            className="btn-secondary flex items-center space-x-2"
          >
            <Pause className="w-4 h-4" />
            <span>{simulationState.paused ? 'Resume' : 'Pause'}</span>
          </button>
          
          <button
            onClick={handleStop}
            disabled={!simulationState.running}
            className="btn-danger flex items-center space-x-2"
          >
            <Square className="w-4 h-4" />
            <span>Stop</span>
          </button>
        </div>
      </div>

      {/* Progress */}
      {simulationState.running && (
        <div className="card">
          <h3 className="text-lg font-medium mb-3">Progress</h3>
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${progressPercentage}%` }}
            ></div>
          </div>
          <div className="mt-2 text-sm text-gray-600">
            {simulationState.progress.completed} / {simulationState.progress.total} conversations completed
            ({progressPercentage.toFixed(1)}%)
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Conversations */}
        <div className="card">
          <h3 className="text-lg font-medium mb-4">Recent Conversations</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {conversationTurns.slice(-10).map((turn, index) => (
              <div
                key={index}
                className={`conversation-turn ${
                  turn.speaker === 'wizard' ? 'turn-wizard' : 'turn-pop'
                }`}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="font-medium text-sm">
                    {turn.speaker === 'wizard' ? 'Wizard' : turn.agent_id || 'Agent'}
                  </span>
                  <span className="text-xs text-gray-500">
                    {new Date(turn.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-sm text-gray-700">{turn.text}</p>
              </div>
            ))}
            {conversationTurns.length === 0 && (
              <p className="text-gray-500 text-center py-8">
                {isDemoMode ? 'Demo conversations will appear here when simulation starts' : 'No conversations yet'}
              </p>
            )}
          </div>
        </div>

        {/* System Events */}
        <div className="card">
          <h3 className="text-lg font-medium mb-4">System Events</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {systemEvents.slice(-10).map((event, index) => (
              <div key={index} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex justify-between items-start mb-1">
                  <span className="font-medium text-sm capitalize">
                    {event.type.replace('_', ' ')}
                  </span>
                  <span className="text-xs text-gray-500">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="text-sm text-gray-600">
                  {typeof event.data === 'string' ? event.data : JSON.stringify(event.data, null, 2)}
                </div>
              </div>
            ))}
            {systemEvents.length === 0 && (
              <p className="text-gray-500 text-center py-8">No events yet</p>
            )}
          </div>
        </div>
      </div>

      {/* Recent Runs Summary */}
      {runs && runs.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-medium mb-4">Recent Simulation Runs</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Run ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Population
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Progress
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {runs.slice(0, 5).map((run) => (
                  <tr key={run.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      #{run.id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        run.status === 'completed' 
                          ? 'bg-green-100 text-green-800'
                          : run.status === 'running'
                          ? 'bg-blue-100 text-blue-800'
                          : run.status === 'failed'
                          ? 'bg-red-100 text-red-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {run.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {run.population_size}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {run.progress}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(run.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default Dashboard