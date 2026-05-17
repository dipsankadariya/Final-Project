import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { LoginCard } from './GoogleLogin'
import { Landing } from './Landing'
import { About } from './About'
import { Chat } from './Chat'
import { Forum } from './Forum'
import { ForumDetail } from './ForumDetail'

export default function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  // Load user info from localStorage on mount
  useEffect(() => {
    const token = localStorage.getItem('auth_token')
    const userInfo = localStorage.getItem('user_info')
    if (token && userInfo) {
      try {
        setUser(JSON.parse(userInfo))
      } catch (e) {
        console.error('Failed to parse user info:', e)
        localStorage.removeItem('auth_token')
        localStorage.removeItem('user_info')
      }
    }
    setLoading(false)
  }, [])

  const handleLoginSuccess = (data) => {
    setUser(data.user)
  }

  const handleLogout = () => {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('user_info')
    setUser(null)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin" />
      </div>
    )
  }

  // If not authenticated, only show login page
  if (!user) {
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginCard onLoginSuccess={handleLoginSuccess} />} />
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </BrowserRouter>
    )
  }

  // If authenticated, show full navigation (home, about, chat)
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/about" element={<About />} />
        <Route path="/chat" element={<Chat user={user} onLogout={handleLogout} />} />
        <Route path="/forum" element={<Forum user={user} onLogout={handleLogout} />} />
        <Route path="/forum/:id" element={<ForumDetail user={user} onLogout={handleLogout} />} />
        <Route path="/login" element={<Navigate to="/" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}