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
      <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg text-center">
        <p className="text-sm text-yellow-800 font-medium">⚠️ Google authentication not configured</p>
        <p className="text-xs text-yellow-600 mt-1">Set VITE_GOOGLE_CLIENT_ID in your .env.local file</p>
      </div>
    )
  }

  return (
    <div className="flex justify-center">
      {isLoading && (
        <div className="w-full h-12 bg-slate-100 rounded-lg animate-pulse" />
      )}
      <div ref={buttonRef} style={{ minHeight: isLoading ? '0' : 'auto' }} />
    </div>
  )
}

export function LoginCard({ onLoginSuccess }) {
  const [error, setError] = useState(null)

  return (
    <div className="min-h-screen bg-white flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-12 h-12 bg-slate-100 rounded-lg mb-4">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" className="text-slate-700">
              <path d="M12 1L3 5v6c0 5.25 3.75 10.15 9 11.35C17.25 21.15 21 16.25 21 11V5L12 1z" />
            </svg>
          </div>
          <h1 className="text-3xl font-semibold text-slate-900 mb-2">नेपाली Legal QA</h1>
          <p className="text-slate-500">Sign in to your account</p>
        </div>

        {/* Login card */}
        <div className="border border-slate-200 rounded-lg p-8 space-y-6">
          <GoogleLoginButton
            onLoginSuccess={onLoginSuccess}
            onLoginError={(err) => setError(err)}
          />

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-slate-200" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white text-slate-500">or continue as guest</span>
            </div>
          </div>

          <button
            onClick={() => {
              localStorage.setItem('auth_token', '')
              localStorage.setItem('user_info', JSON.stringify({ name: 'Guest', email: '', sub: 'guest', picture: null }))
              onLoginSuccess?.({ user: { name: 'Guest', email: '', sub: 'guest', picture: null } })
            }}
            className="w-full py-2 px-4 border border-slate-200 rounded-lg text-slate-700 hover:bg-slate-50 transition-colors text-sm font-medium"
          >
            Continue as Guest
          </button>

          <p className="text-center text-xs text-slate-500">
            We use Google to keep your information secure.
          </p>
        </div>
      </div>
    </div>
  )
}

export function UserProfile({ user, onLogout }) {
  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-purple-50 rounded-lg border border-purple-200">
      {user.picture && (
        <img
          src={user.picture}
          alt={user.name}
          className="w-8 h-8 rounded-full"
        />
      )}
      <div className="flex-1">
        <p className="text-sm font-medium text-gray-900">{user.name}</p>
        <p className="text-xs text-gray-500">{user.email}</p>
      </div>
      <button
        onClick={onLogout}
        className="text-xs px-3 py-1 rounded bg-white border border-purple-200 text-purple-700 hover:bg-purple-50 transition-colors"
      >
        Logout
      </button>
    </div>
  )
}
