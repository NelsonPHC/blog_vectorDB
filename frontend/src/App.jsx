import React, { useState } from 'react'
import Search from './Search'
import Chat from './Chat'
import './index.css'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [activeTab, setActiveTab] = useState('search')

  return (
    <div className="container">
      <div className="header">
        <h1>🔍 Vector Search System</h1>
        <p>Search blog posts using semantic similarity</p>
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          Search
        </button>
        <button
          className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          Chat (RAG)
        </button>
      </div>

      {activeTab === 'search' ? <Search apiUrl={API_BASE_URL} /> : <Chat apiUrl={API_BASE_URL} />}
    </div>
  )
}

export default App

