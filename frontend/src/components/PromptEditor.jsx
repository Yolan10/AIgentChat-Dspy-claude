import React, { useState, useEffect } from 'react';
import { Save, RefreshCw, FileText } from 'lucide-react';

const PromptEditor = () => {
  const [templates, setTemplates] = useState({});
  const [activeTemplate, setActiveTemplate] = useState('wizard_prompt.txt');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchTemplates();
  }, []);

  const fetchTemplates = async () => {
    try {
      const response = await fetch('/api/templates');
      const data = await response.json();
      setTemplates(data);
    } catch (error) {
      console.error('Error fetching templates:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const response = await fetch(`/api/templates/${activeTemplate}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: templates[activeTemplate] })
      });
      const result = await response.json();
      if (response.ok) {
        alert('Template saved successfully!');
      } else {
        alert('Error saving template: ' + result.error);
      }
    } catch (error) {
      alert('Error saving template: ' + error.message);
    } finally {
      setSaving(false);
    }
  };

  const handleTemplateChange = (content) => {
    setTemplates(prev => ({ ...prev, [activeTemplate]: content }));
  };

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="w-6 h-6 animate-spin" />
          <span className="ml-2">Loading templates...</span>
        </div>
      </div>
    );
  }

  const templateInfo = {
    'wizard_prompt.txt': {
      title: 'Wizard Prompt',
      description: 'The main prompt template for the wizard agent. Use {{goal}} placeholder for the goal.'
    },
    'judge_prompt.txt': {
      title: 'Judge Prompt',
      description: 'Template for the judge agent that evaluates conversations. Use {{goal}} and {{transcript}} placeholders.'
    },
    'population_instruction.txt': {
      title: 'Population Instruction',
      description: 'Template for generating population agents. Use {{instruction}} and {{n}} placeholders.'
    },
    'self_improve_prompt.txt': {
      title: 'Self-Improvement Prompt',
      description: 'Template for the wizard\'s self-improvement process. Use {{logs}} placeholder.'
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Prompt Editor</h2>
        <div className="flex space-x-3">
          <button
            onClick={fetchTemplates}
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

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Template List */}
        <div className="card">
          <h3 className="text-lg font-medium mb-4">Templates</h3>
          <div className="space-y-2">
            {Object.keys(templates).map((templateName) => (
              <button
                key={templateName}
                onClick={() => setActiveTemplate(templateName)}
                className={`w-full text-left p-3 rounded-lg border transition-colors ${
                  activeTemplate === templateName
                    ? 'bg-primary-50 border-primary-200 text-primary-700'
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <FileText className="w-4 h-4" />
                  <span className="font-medium">
                    {templateInfo[templateName]?.title || templateName}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Template Editor */}
        <div className="lg:col-span-3 card">
          {activeTemplate && (
            <>
              <div className="mb-4">
                <h3 className="text-lg font-medium">
                  {templateInfo[activeTemplate]?.title || activeTemplate}
                </h3>
                <p className="text-sm text-gray-600 mt-1">
                  {templateInfo[activeTemplate]?.description}
                </p>
              </div>
              
              <textarea
                value={templates[activeTemplate] || ''}
                onChange={(e) => handleTemplateChange(e.target.value)}
                className="textarea w-full h-96 font-mono text-sm"
                placeholder="Enter your prompt template here..."
              />
              
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <h4 className="font-medium text-sm mb-2">Available Placeholders:</h4>
                <div className="text-sm text-gray-600">
                  {activeTemplate === 'wizard_prompt.txt' && (
                    <span className="inline-block bg-gray-200 px-2 py-1 rounded mr-2">{{goal}}</span>
                  )}
                  {activeTemplate === 'judge_prompt.txt' && (
                    <>
                      <span className="inline-block bg-gray-200 px-2 py-1 rounded mr-2">{{goal}}</span>
                      <span className="inline-block bg-gray-200 px-2 py-1 rounded mr-2">{{transcript}}</span>
                    </>
                  )}
                  {activeTemplate === 'population_instruction.txt' && (
                    <>
                      <span className="inline-block bg-gray-200 px-2 py-1 rounded mr-2">{{instruction}}</span>
                      <span className="inline-block bg-gray-200 px-2 py-1 rounded mr-2">{{n}}</span>
                    </>
                  )}
                  {activeTemplate === 'self_improve_prompt.txt' && (
                    <span className="inline-block bg-gray-200 px-2 py-1 rounded mr-2">{{logs}}</span>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default PromptEditor;