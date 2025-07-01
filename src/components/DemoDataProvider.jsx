import React, { createContext, useContext, useState, useEffect } from 'react'
import { config } from '@/lib/config'

const DemoDataContext = createContext({})

export const useDemoData = () => {
  const context = useContext(DemoDataContext)
  if (!context) {
    throw new Error('useDemoData must be used within a DemoDataProvider')
  }
  return context
}

// Mock data for demo purposes
const generateMockData = () => {
  const runs = [
    {
      id: '1',
      created_at: new Date(Date.now() - 86400000).toISOString(),
      instruction: 'Generate diverse hearing loss population',
      population_size: 36,
      goal: 'Uncover key insights about hearing loss experiences',
      status: 'completed',
      progress: 100,
    },
    {
      id: '2', 
      created_at: new Date(Date.now() - 172800000).toISOString(),
      instruction: 'Create personas with varying hearing aid experience',
      population_size: 24,
      goal: 'Research hearing aid adoption barriers',
      status: 'completed',
      progress: 100,
    },
    {
      id: '3',
      created_at: new Date().toISOString(),
      instruction: 'Generate population for workplace study',
      population_size: 12,
      goal: 'Understand workplace accommodation needs',
      status: 'running',
      progress: 67,
    }
  ]

  const agents = runs.flatMap((run, runIndex) => 
    Array.from({ length: Math.min(run.population_size, 5) }, (_, i) => ({
      id: `${run.id}-agent-${i}`,
      run_id: run.id,
      agent_id: `${runIndex + 1}.${i + 1}`,
      name: ['Emma Carter', 'Michael Thompson', 'Sarah Williams', 'David Johnson', 'Linda Martinez'][i],
      personality: `O:${(0.3 + Math.random() * 0.7).toFixed(1)} C:${(0.3 + Math.random() * 0.7).toFixed(1)} E:${(0.2 + Math.random() * 0.6).toFixed(1)} A:${(0.4 + Math.random() * 0.5).toFixed(1)} N:${(0.2 + Math.random() * 0.5).toFixed(1)}`,
      age: 25 + Math.floor(Math.random() * 50),
      occupation: ['teacher', 'engineer', 'nurse', 'accountant', 'manager'][i],
      initial_goals: 'improve communication in noisy environments',
      memory_summary: 'struggled with hearing loss for several years',
    }))
  )

  const conversations = agents.map(agent => ({
    id: `conv-${agent.id}`,
    run_id: agent.run_id,
    agent_id: agent.agent_id,
    wizard_id: 'Wizard_001',
    turns: [
      { speaker: 'wizard', text: 'Hello! Thank you for participating in our research study.' },
      { speaker: 'pop', text: `Hi, I'm ${agent.name}. Happy to help with your hearing loss research.` },
      { speaker: 'wizard', text: 'Can you tell me about your experience with hearing loss?' },
      { speaker: 'pop', text: 'It has been challenging, especially in social situations and at work.' },
    ],
    status: 'completed',
  }))

  const evaluations = conversations.map(conv => ({
    id: `eval-${conv.id}`,
    conversation_id: conv.id,
    judge_id: 'Judge_001',
    goal_completion: 0.7 + Math.random() * 0.3,
    coherence: 0.8 + Math.random() * 0.2,
    tone: 0.75 + Math.random() * 0.25,
    overall_score: 0.7 + Math.random() * 0.3,
    success: Math.random() > 0.3,
    rationale: 'Good conversation flow with comprehensive research plan generated.',
    confidence: 0.8,
  }))

  const tokenUsage = runs.map(run => ({
    id: `token-${run.id}`,
    run_id: run.id,
    model: 'gpt-4.1-nano',
    prompt_tokens: 1500 + Math.floor(Math.random() * 1000),
    completion_tokens: 800 + Math.floor(Math.random() * 500),
    total_tokens: 2300 + Math.floor(Math.random() * 1500),
    cost: (0.05 + Math.random() * 0.1).toFixed(4),
  }))

  return { runs, agents, conversations, evaluations, tokenUsage }
}

export function DemoDataProvider({ children }) {
  const [demoData, setDemoData] = useState(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simulate loading time
    const timer = setTimeout(() => {
      setDemoData(generateMockData())
      setIsLoading(false)
    }, 1000)

    return () => clearTimeout(timer)
  }, [])

  const value = {
    ...demoData,
    isLoading,
    isDemoMode: !config.supabase.url || !config.supabase.anonKey,
  }

  return (
    <DemoDataContext.Provider value={value}>
      {children}
    </DemoDataContext.Provider>
  )
}