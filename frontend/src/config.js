// API configuration for different environments
const API_BASE_URL = process.env.REACT_APP_API_URL || 
  (process.env.NODE_ENV === 'production' 
    ? 'https://aigentchat-dspy-pf7r.onrender.com/'  // Replace with your actual Render backend URL
    : 'http://localhost:5000');

const WS_URL = process.env.REACT_APP_WS_URL || API_BASE_URL;

export const config = {
  API_BASE_URL,
  WS_URL,
  API_ENDPOINTS: {
    login: `${API_BASE_URL}/api/login`,
    logout: `${API_BASE_URL}/api/logout`,
    checkAuth: `${API_BASE_URL}/api/check_auth`,
    status: `${API_BASE_URL}/api/status`,
    config: `${API_BASE_URL}/api/config`,
    templates: `${API_BASE_URL}/api/templates`,
    run: `${API_BASE_URL}/api/run`,
    pause: `${API_BASE_URL}/api/pause`,
    stop: `${API_BASE_URL}/api/stop`,
    logs: {
      summary: (runNo) => `${API_BASE_URL}/api/logs/summary/${runNo}`,
      scores: `${API_BASE_URL}/api/logs/scores`,
      runs: `${API_BASE_URL}/api/logs/runs`,
      tokenUsage: `${API_BASE_URL}/api/logs/token_usage`,
      summarize: (filename) => `${API_BASE_URL}/api/logs/summarize/${filename}`,
      lengths: `${API_BASE_URL}/api/logs/lengths`,
      agentStats: `${API_BASE_URL}/api/logs/agent_stats`,
      conversation: (filename) => `${API_BASE_URL}/api/logs/conversation/${filename}`,
      search: `${API_BASE_URL}/api/search_logs`,
      metricsCsv: `${API_BASE_URL}/api/logs/metrics.csv`
    },
    judge: {
      performance: `${API_BASE_URL}/api/judge/performance`,
      metrics: (judgeId) => `${API_BASE_URL}/api/judge/${judgeId}/metrics`,
      feedback: (judgeId) => `${API_BASE_URL}/api/judge/${judgeId}/feedback`
    }
  }
};

export default config;
