import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter } from 'recharts';
import { RefreshCw, TrendingUp, Target, Award, Download } from 'lucide-react';

const EffectivenessChart = () => {
  const [scores, setScores] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedRun, setSelectedRun] = useState('all');
  const [runs, setRuns] = useState([]);
  const [lengths, setLengths] = useState([]);
  const [agentStats, setAgentStats] = useState([]);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [scoresResponse, runsResponse, lengthsRes, agentsRes] = await Promise.all([
        fetch('/api/logs/scores'),
        fetch('/api/logs/runs'),
        fetch('/api/logs/lengths'),
        fetch('/api/logs/agent_stats')
      ]);

      const scoresData = await scoresResponse.json();
      const runsData = await runsResponse.json();
      const lengthsData = await lengthsRes.json();
      const agentData = await agentsRes.json();

      setScores(scoresData);
      setRuns(runsData);
      setLengths(lengthsData);
      setAgentStats(agentData);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredScores = selectedRun === 'all' 
    ? scores 
    : scores.filter(score => score.run === parseInt(selectedRun));

  const chartData = filteredScores.map((score, index) => ({
    conversation: score.conversation,
    score: score.score,
    improved: score.improved,
    run: score.run,
    index: index + 1
  }));

  const runStats = runs.map(runNo => {
    const runScores = scores.filter(s => s.run === runNo);
    const avgScore = runScores.length > 0 
      ? runScores.reduce((sum, s) => sum + s.score, 0) / runScores.length 
      : 0;
    const improvements = runScores.filter(s => s.improved).length;
    const successRate = runScores.filter(s => s.score > 0.5).length / runScores.length * 100;
    
    return {
      run: runNo,
      avgScore: avgScore.toFixed(3),
      improvements,
      conversations: runScores.length,
      successRate: successRate.toFixed(1)
    };
  });

  const lengthDistribution = React.useMemo(() => {
    const counts = {};
    lengths.forEach(l => {
      counts[l.length] = (counts[l.length] || 0) + 1;
    });
    return Object.keys(counts).sort((a, b) => a - b).map(len => ({
      length: parseInt(len),
      count: counts[len]
    }));
  }, [lengths]);

  const scatterData = lengths.map(l => ({
    length: l.length,
    score: l.score
  }));

  const overallStats = {
    totalConversations: scores.length,
    avgScore: scores.length > 0 ? (scores.reduce((sum, s) => sum + s.score, 0) / scores.length).toFixed(3) : 0,
    totalImprovements: scores.filter(s => s.improved).length,
    successRate: scores.length > 0 ? (scores.filter(s => s.score > 0.5).length / scores.length * 100).toFixed(1) : 0
  };

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="w-6 h-6 animate-spin" />
          <span className="ml-2">Loading effectiveness data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Wizard Effectiveness</h2>
        <div className="flex space-x-3">
          <select
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
            className="input w-auto"
          >
            <option value="all">All Runs</option>
            {runs.map(run => (
              <option key={run} value={run}>Run {run}</option>
            ))}
          </select>
          <button
            onClick={fetchData}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
          <button
            onClick={async () => {
              const resp = await fetch('/api/logs/metrics.csv');
              const text = await resp.text();
              const blob = new Blob([text], { type: 'text/csv' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = 'metrics.csv';
              link.click();
              URL.revokeObjectURL(url);
            }}
            className="btn-secondary flex items-center space-x-2"
          >
            <Download className="w-4 h-4" />
            <span>CSV</span>
          </button>
        </div>
      </div>

      {/* Overall Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Target className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Conversations</p>
              <p className="text-2xl font-bold">{overallStats.totalConversations}</p>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Average Score</p>
              <p className="text-2xl font-bold">{overallStats.avgScore}</p>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Award className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Success Rate</p>
              <p className="text-2xl font-bold">{overallStats.successRate}%</p>
            </div>
          </div>
        </div>
        
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-orange-100 rounded-lg">
              <RefreshCw className="w-6 h-6 text-orange-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Improvements</p>
              <p className="text-2xl font-bold">{overallStats.totalImprovements}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Score Trend Chart */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">Score Trend Over Time</h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="index" 
              label={{ value: 'Conversation Index', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              domain={[0, 1]}
              label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip 
              formatter={(value, name) => [value.toFixed(3), name]}
              labelFormatter={(label) => `Conversation ${label}`}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="score" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Run Comparison */}
      {runs.length > 1 && (
        <div className="card">
          <h3 className="text-lg font-medium mb-4">Run Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={runStats}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="run" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="avgScore" fill="#3b82f6" name="Average Score" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Conversation Length Distribution */}
      {lengthDistribution.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-medium mb-4">Conversation Lengths</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={lengthDistribution}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="length" label={{ value: 'Turns', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Bar dataKey="count" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Score vs Length */}
      {scatterData.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-medium mb-4">Score vs Length</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid />
              <XAxis dataKey="length" type="number" name="Turns" />
              <YAxis dataKey="score" type="number" domain={[0,1]} name="Score" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter data={scatterData} fill="#e11d48" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Detailed Run Stats */}
      <div className="card">
        <h3 className="text-lg font-medium mb-4">Run Statistics</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Run
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Conversations
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Average Score
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Success Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Improvements
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {runStats.map((stat) => (
                <tr key={stat.run}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    Run {stat.run}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {stat.conversations}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {stat.avgScore}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {stat.successRate}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {stat.improvements}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default EffectivenessChart;