import { useEffect, useRef, useState } from 'react'

export function GoogleLoginButton({ onLoginSuccess, onLoginError }) {
  const buttonRef = useRef(null)
  const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) {
      console.warn('VITE_GOOGLE_CLIENT_ID not set in environment')
      setIsLoading(false)
      return
    }

    // Load Google Identity Services Library
    const script = document.createElement('script')
    script.src = 'https://accounts.google.com/gsi/client'
    script.async = true
    script.defer = true

    script.onload = () => {
      if (window.google && buttonRef.current) {
        window.google.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: handleCredentialResponse,
        })

        window.google.accounts.id.renderButton(buttonRef.current, {
          theme: 'outline',
          size: 'large',
          locale: 'en',
          width: '400',
        })
        setIsLoading(false)
      }
    }

    script.onerror = () => {
      console.error('Failed to load Google Identity Services')
      setIsLoading(false)
    }

    document.head.appendChild(script)

    return () => {
      try {
        document.head.removeChild(script)
      } catch (e) {}
    }
  }, [GOOGLE_CLIENT_ID])

  const handleCredentialResponse = async (response) => {
    try {
      const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')
      const endpoint = API_BASE ? `${API_BASE}/api/auth/google` : '/api/auth/google'

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ token: response.credential }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        onLoginError?.(err.detail ?? `HTTP ${res.status}`)
        return
      }

      const data = await res.json()
      localStorage.setItem('auth_token', data.access_token)
      localStorage.setItem('user_info', JSON.stringify(data.user))
      onLoginSuccess?.(data)
    } catch (error) {
      console.error('Login error:', error)
      onLoginError?.(error.message)
    }
  }

  if (!GOOGLE_CLIENT_ID) {
    return (
      <div className="p-4 bg-amber-50 border border-amber-200 rounded-md text-center">
        <p className="text-sm text-amber-800 font-medium">⚠️ Google authentication not configured</p>
        <p className="text-xs text-amber-700 mt-1">Set VITE_GOOGLE_CLIENT_ID in your .env.local file</p>
      </div>
    )
  }

  return (
    <div className="flex justify-center">
      {isLoading && (
        <div className="w-full h-11 bg-gray-200 rounded-md animate-pulse" />
      )}
      <div ref={buttonRef} style={{ minHeight: isLoading ? '0' : 'auto' }} />
    </div>
  )
}

export function LoginCard({ onLoginSuccess }) {
  const [error, setError] = useState(null)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex">
      {/* Left Side - Visual Section */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-purple-600 via-purple-700 to-indigo-800 relative overflow-hidden flex-col items-center justify-center p-12">
        {/* Decorative Elements */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-indigo-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse delay-2000"></div>
        
        {/* Content */}
        <div className="relative z-10 text-center max-w-md">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-white/20 backdrop-blur-md rounded-2xl text-white mb-8 border border-white/30">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 1L3 5v6c0 5.25 3.75 10.15 9 11.35C17.25 21.15 21 16.25 21 11V5L12 1z" />
            </svg>
          </div>
          
          <h2 className="text-4xl font-bold text-white mb-4">न्याय</h2>
          <p className="text-lg text-purple-100 mb-8">Your AI-powered guide to Nepali legal insights</p>
          
          {/* Feature List */}
          <div className="space-y-4 text-left">
            <div className="flex items-start gap-3">
              <div className="mt-1 w-5 h-5 rounded-full bg-white/30 flex items-center justify-center flex-shrink-0">
                <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <p className="text-white font-medium">Instant Legal Answers</p>
                <p className="text-sm text-purple-200">Get answers to legal questions 24/7</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="mt-1 w-5 h-5 rounded-full bg-white/30 flex items-center justify-center flex-shrink-0">
                <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <p className="text-white font-medium">Nepali Legal Context</p>
                <p className="text-sm text-purple-200">Guidance specific to Nepal</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="mt-1 w-5 h-5 rounded-full bg-white/30 flex items-center justify-center flex-shrink-0">
                <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <p className="text-white font-medium">Conversational Interface</p>
                <p className="text-sm text-purple-200">Chat naturally about legal matters</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Login Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center px-6 sm:px-12">
        <div className="w-full max-w-sm">
          {/* Header */}
          <div className="mb-10">
            <div className="lg:hidden flex items-center gap-3 mb-8">
              <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-br from-purple-600 to-indigo-600 rounded-lg text-white">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 1L3 5v6c0 5.25 3.75 10.15 9 11.35C17.25 21.15 21 16.25 21 11V5L12 1z" />
                </svg>
              </div>
              <h1 className="text-2xl font-bold text-gray-900">न्याय</h1>
            </div>
            
            <h2 className="text-3xl font-bold text-gray-900 mb-2">Welcome Back</h2>
            <p className="text-gray-600">Sign in to your account to continue</p>
          </div>

          {/* Login Form */}
          <div className="space-y-5">
            {/* Google Login Button */}
            <GoogleLoginButton
              onLoginSuccess={onLoginSuccess}
              onLoginError={(err) => setError(err)}
            />

            {/* Error Message */}
            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg animate-in fade-in">
                <p className="text-sm text-red-700 font-medium">Sign in failed</p>
                <p className="text-sm text-red-600 mt-1">{error}</p>
              </div>
            )}

            {/* Divider */}
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-200" />
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-3 bg-gradient-to-br from-slate-50 to-slate-100 text-gray-500 font-medium">Or</span>
              </div>
            </div>

            {/* Guest Button */}
            <button
              onClick={() => {
                localStorage.setItem('auth_token', '')
                localStorage.setItem('user_info', JSON.stringify({ name: 'Guest', email: '', sub: 'guest', picture: null }))
                onLoginSuccess?.({ user: { name: 'Guest', email: '', sub: 'guest', picture: null } })
              }}
              className="w-full px-4 py-3 text-sm font-semibold text-gray-700 border-2 border-gray-200 rounded-lg hover:bg-gray-50 hover:border-gray-300 transition-all duration-200 active:scale-95"
            >
              Continue as Guest
            </button>

            {/* Footer Text */}
            <p className="text-center text-xs text-gray-500 pt-2">
              By signing in, you agree to our{' '}
              <a href="#" className="text-purple-600 font-semibold hover:underline">
                Terms
              </a>
              {' '}and{' '}
              <a href="#" className="text-purple-600 font-semibold hover:underline">
                Privacy
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export function UserProfile({ user, onLogout }) {
  return (
    <div className="flex items-center gap-3">
      {user.picture && (
        <img
          src={user.picture}
          alt={user.name}
          className="w-9 h-9 rounded-full border border-gray-200"
        />
      )}
      <div className="flex-1 min-w-0 hidden sm:block">
        <p className="text-sm font-medium text-gray-900 truncate">{user.name}</p>
        {user.email && <p className="text-xs text-gray-500 truncate">{user.email}</p>}
      </div>
      <button
        onClick={onLogout}
        className="text-xs px-3 py-1.5 rounded-md border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors whitespace-nowrap font-medium"
      >
        Logout
      </button>
    </div>
  )
}
