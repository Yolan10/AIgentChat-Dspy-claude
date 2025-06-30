import React, { useState, useEffect } from 'react';
import { RefreshCw } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

const TokenUsage = () => {
  const [usage, setUsage] = useState({});
  const [loading, setLoading] = useState(true);

  const fetchUsage = async () => {
    try {
      const res = await fetch('/api/logs/token_usage');
      const data = await res.json();
      setUsage(data);
    } catch (err) {
      console.error('Error fetching token usage:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsage();
  }, []);

  const usageData = Object.entries(usage).map(([run, stats]) => ({
    run,
    prompt_tokens: stats.prompt_tokens,
    completion_tokens: stats.completion_tokens,
    cost: stats.cost
  }));

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="w-6 h-6 animate-spin" />
          <span className="ml-2">Loading token usage...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Token Usage</h2>
        <button onClick={fetchUsage} className="btn-secondary flex items-center space-x-2">
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      <div className="card">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={usageData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="run" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="prompt_tokens" fill="#3b82f6" name="Prompt Tokens" />
            <Bar dataKey="completion_tokens" fill="#10b981" name="Completion Tokens" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Run</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Prompt Tokens</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Completion Tokens</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Cost ($)</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {usageData.map(row => (
              <tr key={row.run}>
                <td className="px-6 py-4 whitespace-nowrap text-sm">Run {row.run}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">{row.prompt_tokens}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">{row.completion_tokens}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">{row.cost.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default TokenUsage;

