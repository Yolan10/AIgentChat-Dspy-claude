import React, { useState } from 'react';
import { Play, Pause, Square, RefreshCw } from 'lucide-react';
import { config } from '../config';

const Dashboard = ({ simulationState, conversationTurns, systemEvents }) => {
  const [formData, setFormData] = useState({
    instruction: 'Generate population',
    population_size: 36,
    goal: 'Uncover key insights about hearing loss experiences and generate a structured research plan'
  });

  const handleStart = async () => {
    try {
      const response = await fetch(config.API_ENDPOINTS.run, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(formData)
      });
      const result = await response.json();
      if (!response.ok) {
        alert(result.error || 'Failed to start simulation');
      }
    } catch (error) {
      alert('Error starting simulation: ' + error.message);
    }
  };

  const handlePause = async () => {
    try {
      const response = await fetch(config.API_ENDPOINTS.pause, { 
        method: 'POST',
        credentials: 'include'
      });
      const result = await response.json();
      if (!response.ok) {
        alert(result.error || 'Failed to pause simulation');
      }
    } catch (error) {
      alert('Error pausing simulation: ' + error.message);
    }
  };

  const handleStop = async () => {
    try {
      const response = await fetch(config.API_ENDPOINTS.stop, { 
        method: 'POST',
        credentials: 'include'
      });
      const result = await response.json();
      if (!response.ok) {
        alert(result.error || 'Failed to stop simulation');
      }
    } catch (error) {
      alert('Error stopping simulation: ' + error.message);
    }
  };

  const progressPercentage = simulationState.progress.total > 0 
    ? (simulationState.progress.completed / simulationState.progress.total) * 100 
    : 0;

  return (
    <div className="space-y-6">
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
              <p className="text-gray-500 text-center py-8">No conversations yet</p>
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
                  {JSON.stringify(event.data, null, 2)}
                </div>
              </div>
            ))}
            {systemEvents.length === 0 && (
              <p className="text-gray-500 text-center py-8">No events yet</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
