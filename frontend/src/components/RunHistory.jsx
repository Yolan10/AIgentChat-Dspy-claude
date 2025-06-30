import React, { useState, useEffect } from 'react';
import { RefreshCw, Download, Eye, Calendar, Users, Target } from 'lucide-react';

const RunHistory = () => {
  const [runs, setRuns] = useState([]);
  const [summaries, setSummaries] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedRun, setSelectedRun] = useState(null);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [transcript, setTranscript] = useState([]);
  const [convSummary, setConvSummary] = useState('');
  const [loadingConv, setLoadingConv] = useState(false);

  useEffect(() => {
    fetchRuns();
  }, []);

  const fetchRuns = async () => {
    try {
      const response = await fetch('/api/logs/runs');
      const runsData = await response.json();
      setRuns(runsData);
      
      // Fetch summaries for each run
      const summaryPromises = runsData.map(async (runNo) => {
        try {
          const summaryResponse = await fetch(`/api/logs/summary/${runNo}`);
          if (summaryResponse.ok) {
            const summaryData = await summaryResponse.json();
            return { runNo, summary: summaryData };
          }
        } catch (error) {
          console.error(`Error fetching summary for run ${runNo}:`, error);
        }
        return { runNo, summary: null };
      });
      
      const summaryResults = await Promise.all(summaryPromises);
      const summariesMap = {};
      summaryResults.forEach(({ runNo, summary }) => {
        summariesMap[runNo] = summary;
      });
      setSummaries(summariesMap);
    } catch (error) {
      console.error('Error fetching runs:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRunStats = (summary) => {
    if (!summary || !Array.isArray(summary)) return null;
    
    const totalAgents = summary.length;
    const successfulConversations = summary.filter(agent => agent.success).length;
    const averageScore = summary.reduce((sum, agent) => sum + (agent.score || 0), 0) / totalAgents;
    const successRate = (successfulConversations / totalAgents) * 100;
    
    return {
      totalAgents,
      successfulConversations,
      averageScore: averageScore.toFixed(3),
      successRate: successRate.toFixed(1)
    };
  };

  const downloadSummary = (runNo) => {
    const summary = summaries[runNo];
    if (!summary) return;
    
    const dataStr = JSON.stringify(summary, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `summary_${runNo}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const loadConversation = async (agent) => {
    setSelectedAgent(agent);
    setLoadingConv(true);
    setTranscript([]);
    setConvSummary('');
    const prefix = `Wizard_001_${agent.pop_agent_id}`;
    try {
      const [logRes, sumRes] = await Promise.all([
        fetch(`/api/logs/conversation/${prefix}`),
        fetch(`/api/logs/summarize/${prefix}`)
      ]);
      if (logRes.ok) {
        const data = await logRes.json();
        setTranscript(data.turns || []);
      }
      if (sumRes.ok) {
        const data = await sumRes.json();
        setConvSummary(data.summary);
      }
    } catch (err) {
      console.error('Error loading conversation:', err);
    } finally {
      setLoadingConv(false);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="w-6 h-6 animate-spin" />
          <span className="ml-2">Loading run history...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Run History</h2>
        <button
          onClick={fetchRuns}
          className="btn-secondary flex items-center space-x-2"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      {runs.length === 0 ? (
        <div className="card text-center py-8">
          <Calendar className="w-12 h-12 mx-auto mb-3 text-gray-300" />
          <p className="text-gray-500">No simulation runs found</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Run List */}
          <div className="card">
            <h3 className="text-lg font-medium mb-4">Available Runs</h3>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {runs.map((runNo) => {
                const summary = summaries[runNo];
                const stats = getRunStats(summary);
                
                return (
                  <div
                    key={runNo}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedRun === runNo
                        ? 'border-primary-300 bg-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedRun(runNo)}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-medium">Run #{runNo}</h4>
                      <div className="flex space-x-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            downloadSummary(runNo);
                          }}
                          className="p-1 text-gray-400 hover:text-gray-600"
                          title="Download Summary"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedRun(runNo);
                          }}
                          className="p-1 text-gray-400 hover:text-gray-600"
                          title="View Details"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    
                    {stats && (
                      <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
                        <div className="flex items-center space-x-1">
                          <Users className="w-3 h-3" />
                          <span>{stats.totalAgents} agents</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Target className="w-3 h-3" />
                          <span>{stats.successRate}% success</span>
                        </div>
                        <div className="col-span-2">
                          Avg Score: {stats.averageScore}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Run Details */}
          <div className="card">
            <h3 className="text-lg font-medium mb-4">
              {selectedRun ? `Run #${selectedRun} Details` : 'Select a Run'}
            </h3>
            
            {selectedRun && summaries[selectedRun] ? (
              <div className="space-y-4">
                {/* Summary Stats */}
                {(() => {
                  const stats = getRunStats(summaries[selectedRun]);
                  return stats ? (
                    <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
                      <div>
                        <p className="text-sm text-gray-600">Total Agents</p>
                        <p className="text-xl font-bold">{stats.totalAgents}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Success Rate</p>
                        <p className="text-xl font-bold">{stats.successRate}%</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Successful</p>
                        <p className="text-xl font-bold">{stats.successfulConversations}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Avg Score</p>
                        <p className="text-xl font-bold">{stats.averageScore}</p>
                      </div>
                    </div>
                  ) : null;
                })()}
                
                {/* Agent List */}
                <div className="max-h-64 overflow-y-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                          Agent
                        </th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                          Score
                        </th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                          Success
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {summaries[selectedRun].map((agent, index) => (
                        <tr key={index} className="cursor-pointer hover:bg-gray-50" onClick={() => loadConversation(agent)}>
                          <td className="px-3 py-2 text-sm">
                            <div>
                              <p className="font-medium">{agent.name}</p>
                              <p className="text-gray-500 text-xs">{agent.pop_agent_id}</p>
                            </div>
                          </td>
                          <td className="px-3 py-2 text-sm">
                            {(agent.score || 0).toFixed(3)}
                          </td>
                          <td className="px-3 py-2 text-sm">
                            <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              agent.success 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-red-100 text-red-800'
                            }`}>
                              {agent.success ? 'Yes' : 'No'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {selectedAgent && (
                  <div className="mt-4 p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Conversation with {selectedAgent.name}</h4>
                    {loadingConv ? (
                      <p>Loading...</p>
                    ) : (
                      <>
                        {convSummary && (
                          <div className="mb-3 p-3 bg-gray-50 rounded">
                            <p className="font-semibold mb-1">Summary</p>
                            <p className="text-sm whitespace-pre-line">{convSummary}</p>
                          </div>
                        )}
                        <div className="max-h-64 overflow-y-auto space-y-2 text-sm">
                          {transcript.map((t, i) => (
                            <div key={i} className={t.speaker === 'wizard' ? 'text-blue-700' : 'text-green-700'}>
                              <strong>{t.speaker}:</strong> {t.text}
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            ) : selectedRun ? (
              <div className="text-center py-8 text-gray-500">
                <p>No summary data available for this run</p>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Eye className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                <p>Select a run from the list to view details</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default RunHistory;
