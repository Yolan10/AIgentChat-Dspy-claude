import React, { useState, useEffect } from 'react';
import { MessageSquare, User, Bot, Clock } from 'lucide-react';

const ConversationViewer = ({ conversationTurns }) => {
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [includeLogs, setIncludeLogs] = useState(false);
  const [logResults, setLogResults] = useState([]);

  useEffect(() => {
    if (includeLogs && searchTerm) {
      fetch(`/api/search_logs?q=${encodeURIComponent(searchTerm)}&limit=50`)
        .then(res => res.json())
        .then(setLogResults)
        .catch(() => setLogResults([]));
    } else {
      setLogResults([]);
    }
  }, [searchTerm, includeLogs]);

  const filteredTurns = conversationTurns.filter(turn => {
    const matchesFilter = filter === 'all' || turn.speaker === filter;
    const matchesSearch = searchTerm === '' || 
      turn.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (turn.agent_id && turn.agent_id.toLowerCase().includes(searchTerm.toLowerCase()));
    
    return matchesFilter && matchesSearch;
  });

  const groupedConversations = filteredTurns.reduce((groups, turn) => {
    const key = turn.agent_id || 'unknown';
    if (!groups[key]) {
      groups[key] = [];
    }
    groups[key].push(turn);
    return groups;
  }, {});

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Conversation Viewer</h2>
        <div className="flex space-x-3">
          <input
            type="text"
            placeholder="Search conversations..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input w-64"
          />
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="input w-auto"
          >
            <option value="all">All Speakers</option>
            <option value="wizard">Wizard Only</option>
            <option value="pop">Population Only</option>
          </select>
          <label className="flex items-center space-x-1 text-sm">
            <input
              type="checkbox"
              checked={includeLogs}
              onChange={(e) => setIncludeLogs(e.target.checked)}
            />
            <span>Search logs</span>
          </label>
        </div>
      </div>

      {/* Live Conversation Feed */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4 flex items-center space-x-2">
          <MessageSquare className="w-5 h-5" />
          <span>Live Conversation Feed</span>
        </h3>
        
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {filteredTurns.slice(-20).map((turn, index) => (
            <div
              key={index}
              className={`conversation-turn ${
                turn.speaker === 'wizard' ? 'turn-wizard' : 'turn-pop'
              }`}
            >
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0">
                  {turn.speaker === 'wizard' ? (
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <Bot className="w-4 h-4 text-blue-600" />
                    </div>
                  ) : (
                    <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                      <User className="w-4 h-4 text-green-600" />
                    </div>
                  )}
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 mb-1">
                    <span className="font-medium text-sm">
                      {turn.speaker === 'wizard' ? 'Wizard' : turn.agent_id || 'Agent'}
                    </span>
                    <span className="text-xs text-gray-500 flex items-center space-x-1">
                      <Clock className="w-3 h-3" />
                      <span>{new Date(turn.timestamp).toLocaleTimeString()}</span>
                    </span>
                  </div>
                  <p className="text-sm text-gray-700">{turn.text}</p>
                </div>
              </div>
            </div>
          ))}
          
          {filteredTurns.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <MessageSquare className="w-12 h-12 mx-auto mb-3 text-gray-300" />
              <p>No conversations match your filters</p>
            </div>
          )}
        </div>
      </div>

      {/* Grouped Conversations by Agent */}
      {Object.keys(groupedConversations).length > 0 && (
        <div className="card">
          <h3 className="text-lg font-medium mb-4">Conversations by Agent</h3>
          
          <div className="space-y-4">
            {Object.entries(groupedConversations).map(([agentId, turns]) => (
              <div key={agentId} className="border border-gray-200 rounded-lg p-4">
                <h4 className="font-medium mb-3 flex items-center space-x-2">
                  <User className="w-4 h-4" />
                  <span>{agentId}</span>
                  <span className="text-sm text-gray-500">({turns.length} turns)</span>
                </h4>
                
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {turns.map((turn, index) => (
                    <div
                      key={index}
                      className={`p-2 rounded text-sm ${
                        turn.speaker === 'wizard' 
                          ? 'bg-blue-50 text-blue-800' 
                          : 'bg-green-50 text-green-800'
                      }`}
                    >
                      <div className="font-medium mb-1">
                        {turn.speaker === 'wizard' ? 'Wizard' : 'Agent'}:
                      </div>
                      <div>{turn.text}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {includeLogs && (
        <div className="card">
          <h3 className="text-lg font-medium mb-4 flex items-center space-x-2">
            <Clock className="w-5 h-5" />
            <span>Historical Matches</span>
          </h3>

          <div className="space-y-2 max-h-96 overflow-y-auto">
            {logResults.map((res, index) => (
              <div
                key={index}
                className="border border-gray-200 rounded p-2"
              >
                <div className="text-xs text-gray-500 mb-1">{res.file}</div>
                <div className="text-sm text-gray-700">{res.snippet}</div>
              </div>
            ))}
            {logResults.length === 0 && (
              <div className="text-center text-sm text-gray-500">
                No matches found
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ConversationViewer;