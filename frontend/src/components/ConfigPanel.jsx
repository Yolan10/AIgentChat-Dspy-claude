import React, { useState, useEffect } from 'react';
import { Save, RefreshCw } from 'lucide-react';

const ConfigPanel = () => {
  const [config, setConfig] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await fetch('/api/config');
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error('Error fetching config:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const response = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      const result = await response.json();
      if (response.ok) {
        alert('Configuration saved successfully!');
      } else {
        alert('Error saving configuration: ' + result.error);
      }
    } catch (error) {
      alert('Error saving configuration: ' + error.message);
    } finally {
      setSaving(false);
    }
  };

  const handleChange = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="w-6 h-6 animate-spin" />
          <span className="ml-2">Loading configuration...</span>
        </div>
      </div>
    );
  }

  const configSections = [
    {
      title: 'Population Settings',
      fields: [
        { key: 'POPULATION_SIZE', label: 'Population Size', type: 'number' },
        { key: 'POP_HISTORY_LIMIT', label: 'Population History Limit', type: 'number' }
      ]
    },
    {
      title: 'Wizard Settings',
      fields: [
        { key: 'WIZARD_DEFAULT_GOAL', label: 'Default Goal', type: 'text' },
        { key: 'MAX_TURNS', label: 'Max Turns', type: 'number' },
        { key: 'HISTORY_BUFFER_LIMIT', label: 'History Buffer Limit', type: 'number' }
      ]
    },
    {
      title: 'LLM Settings',
      fields: [
        { key: 'LLM_MODEL', label: 'Model', type: 'text' },
        { key: 'LLM_TEMPERATURE', label: 'Temperature', type: 'number', step: 0.1, min: 0, max: 2 },
        { key: 'LLM_MAX_TOKENS', label: 'Max Tokens', type: 'number' },
        { key: 'LLM_TOP_P', label: 'Top P', type: 'number', step: 0.1, min: 0, max: 1 }
      ]
    },
    {
      title: 'DSPy Settings',
      fields: [
        { key: 'DSPY_TRAINING_ITER', label: 'Training Iterations', type: 'number' },
        { key: 'DSPY_LEARNING_RATE', label: 'Learning Rate', type: 'number', step: 0.01 }
      ]
    },
    {
      title: 'Runtime Options',
      fields: [
        { key: 'SHOW_LIVE_CONVERSATIONS', label: 'Show Live Conversations', type: 'boolean' }
      ]
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Configuration</h2>
        <div className="flex space-x-3">
          <button
            onClick={fetchConfig}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="btn-primary flex items-center space-x-2"
          >
            <Save className="w-4 h-4" />
            <span>{saving ? 'Saving...' : 'Save'}</span>
          </button>
        </div>
      </div>

      {configSections.map((section) => (
        <div key={section.title} className="card">
          <h3 className="text-lg font-medium mb-4">{section.title}</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {section.fields.map((field) => (
              <div key={field.key}>
                <label className="label">{field.label}</label>
                {field.type === 'boolean' ? (
                  <input
                    type="checkbox"
                    checked={config[field.key] || false}
                    onChange={(e) => handleChange(field.key, e.target.checked)}
                    className="w-4 h-4 text-primary-600 bg-gray-100 border-gray-300 rounded focus:ring-primary-500"
                  />
                ) : (
                  <input
                    type={field.type}
                    value={config[field.key] || ''}
                    onChange={(e) => {
                      const value = field.type === 'number' 
                        ? parseFloat(e.target.value) || 0
                        : e.target.value;
                      handleChange(field.key, value);
                    }}
                    step={field.step}
                    min={field.min}
                    max={field.max}
                    className="input"
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="card">
        <h3 className="text-lg font-medium mb-4">Self-Improvement Schedule</h3>
        <div>
          <label className="label">Self Improve After</label>
          <input
            type="text"
            value={Array.isArray(config.SELF_IMPROVE_AFTER) 
              ? config.SELF_IMPROVE_AFTER.join(', ')
              : config.SELF_IMPROVE_AFTER || ''
            }
            onChange={(e) => {
              const value = e.target.value;
              if (value.includes(',')) {
                const array = value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v));
                handleChange('SELF_IMPROVE_AFTER', array);
              } else {
                const num = parseInt(value);
                handleChange('SELF_IMPROVE_AFTER', isNaN(num) ? value : num);
              }
            }}
            placeholder="e.g., 1, 5, 36 or 10"
            className="input"
          />
          <p className="text-sm text-gray-600 mt-1">
            Enter a single number for regular intervals or comma-separated numbers for specific points
          </p>
        </div>
      </div>
    </div>
  );
};

export default ConfigPanel;