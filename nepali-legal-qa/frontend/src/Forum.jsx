import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Scale, RefreshCw, ChevronUp, ChevronDown, MessageSquare, Eye, Send } from 'lucide-react'
import { UserProfile } from './GoogleLogin'

const API_BASE = (import.meta.env.VITE_FORUM_API_BASE ?? '').replace(/\/$/, '')
const FORUM_BASE = API_BASE ? `${API_BASE}/api/forum` : '/api/forum'

async function forumRequest(path, options = {}) {
  const token = localStorage.getItem('auth_token')
  const headers = { 'Content-Type': 'application/json', ...(options.headers ?? {}) }
  if (token) headers.Authorization = `Bearer ${token}`
  const res = await fetch(`${FORUM_BASE}${path}`, { ...options, headers })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail ?? `HTTP ${res.status}`)
  }
  return res.json()
}

function formatRelativeTime(value) {
  const now = Date.now()
  const then = new Date(value).getTime()
  if (Number.isNaN(then)) return 'just now'
  const diffSec = Math.round((now - then) / 1000)
  if (diffSec < 60) return 'just now'
  const diffMin = Math.round(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHours = Math.round(diffMin / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  return `${Math.round(diffHours / 24)}d ago`
}

function getInitials(name) {
  if (!name) return '?'
  const parts = name.trim().split(' ')
  return parts.length >= 2
    ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
    : name.slice(0, 2).toUpperCase()
}

const AVATAR_PALETTE = [
  { bg: '#f3f4f6', text: '#374151' },
  { bg: '#e5e7eb', text: '#1f2937' },
  { bg: '#f9fafb', text: '#4b5563' },
  { bg: '#ececec', text: '#374151' },
  { bg: '#e8e8e8', text: '#1f2937' },
  { bg: '#f1f1f1', text: '#4b5563' },
]

function avatarColor(name = '') {
  const code = [...(name || '')].reduce((acc, c) => acc + c.charCodeAt(0), 0)
  return AVATAR_PALETTE[code % AVATAR_PALETTE.length]
}

function Avatar({ name, size = 34 }) {
  const { bg, text } = avatarColor(name)
  return (
    <div style={{
      width: size, height: size, borderRadius: '50%',
      background: bg, color: text,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: size * 0.35, fontWeight: 600,
      flexShrink: 0, letterSpacing: '0.02em',
      fontFamily: 'inherit',
    }}>
      {getInitials(name)}
    </div>
  )
}

export function Forum({ user, onLogout }) {
  const [posts, setPosts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [title, setTitle] = useState('')
  const [detail, setDetail] = useState('')
  const [posting, setPosting] = useState(false)
  const [postError, setPostError] = useState(null)
  const [votedPosts, setVotedPosts] = useState({})

  const loadPosts = async () => {
    setLoading(true); setError(null)
    try { setPosts(await forumRequest('/questions?sort=new&limit=50')) }
    catch (err) { setError(err.message) }
    finally { setLoading(false) }
  }

  useEffect(() => { loadPosts() }, [])

  const handleCreatePost = async (e) => {
    e.preventDefault()
    const trimmedTitle = title.trim()
    if (!trimmedTitle) { setPostError('Title is required.'); return }
    setPosting(true); setPostError(null)
    try {
      const created = await forumRequest('/questions', {
        method: 'POST',
        body: JSON.stringify({ title: trimmedTitle, body: detail.trim() }),
      })
      setPosts(prev => [created, ...prev])
      setTitle(''); setDetail('')
    } catch (err) { setPostError(err.message) }
    finally { setPosting(false) }
  }

  const handleVote = async (postId, value) => {
    const prev = votedPosts[postId]
    if (prev === value) return
    setVotedPosts(s => ({ ...s, [postId]: value }))
    try {
      await forumRequest(`/questions/${postId}/vote`, { method: 'POST', body: JSON.stringify({ value }) })
      setPosts(prev => prev.map(post => {
        if (post.id !== postId) return post
        let up = post.upvotes, down = post.downvotes
        if (prev === 1) up--; if (prev === -1) down--
        if (value === 1) up++; if (value === -1) down++
        return { ...post, upvotes: up, downvotes: down }
      }))
    } catch (err) {
      setVotedPosts(s => ({ ...s, [postId]: prev }))
      setError(err.message)
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: '#fff', fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}>

      {/* Header */}
      <header style={{ borderBottom: '1px solid #e5e7eb', background: '#fff', position: 'sticky', top: 0, zIndex: 20 }}>
        <div style={{ maxWidth: 1180, margin: '0 auto', padding: '0 24px', height: 52, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
            <div style={{ width: 27, height: 27, borderRadius: 6, background: '#6d28d9', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Scale size={13} color="#fff" />
            </div>
            <span style={{ fontSize: 14, fontWeight: 700, color: '#111827', letterSpacing: '-0.02em' }}>Legal Forum</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Link
              to="/chat"
              style={{ fontSize: 12.5, fontWeight: 600, color: '#6d28d9', border: '1px solid #6d28d9', borderRadius: 6, padding: '5px 14px', textDecoration: 'none', transition: 'background 0.12s' }}
              onMouseEnter={e => { e.currentTarget.style.background = '#6d28d9'; e.currentTarget.style.color = '#fff' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#6d28d9' }}
            >
              Ask AI
            </Link>
            {user && <UserProfile user={user} onLogout={onLogout} />}
          </div>
        </div>
      </header>

      {/* Body */}
      <div style={{ maxWidth: 1180, margin: '0 auto', padding: '0 24px', display: 'grid', gridTemplateColumns: '70% 30%' }}>

        {/* Threads */}
        <section style={{ borderRight: '1px solid #e5e7eb' }}>

          <div style={{ padding: '15px 0 13px', borderBottom: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingRight: 20 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>Threads</span>
            <button
              onClick={loadPosts}
              style={{ display: 'inline-flex', alignItems: 'center', color: '#9ca3af', background: 'none', border: 'none', cursor: 'pointer', padding: 0, transition: 'color 0.12s' }}
              onMouseEnter={e => e.currentTarget.style.color = '#374151'}
              onMouseLeave={e => e.currentTarget.style.color = '#9ca3af'}
            >
              <RefreshCw size={13} />
            </button>
          </div>

          {loading && <p style={{ fontSize: 13, color: '#9ca3af', margin: '20px 0' }}>Loading…</p>}
          {error && <div style={{ fontSize: 12.5, color: '#dc2626', background: '#fef2f2', borderRadius: 6, padding: '10px 12px', margin: '12px 0' }}>{error}</div>}
          {!loading && posts.length === 0 && <p style={{ fontSize: 13, color: '#9ca3af', margin: '20px 0' }}>No threads yet.</p>}

          {posts.map((post) => {
            const voted = votedPosts[post.id]
            const authorName = post.author?.name ?? 'Guest'
            return (
              <article
                key={post.id}
                style={{ borderBottom: '1px solid #f3f4f6', padding: '15px 20px 15px 0', display: 'flex', gap: 12, alignItems: 'flex-start', transition: 'background 0.1s' }}
                onMouseEnter={e => e.currentTarget.style.background = '#fafafa'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                <Avatar name={authorName} size={34} />

                <div style={{ flex: 1, minWidth: 0 }}>

                  {/* Meta */}
                  <div style={{ fontSize: 12.5, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap' }}>
                    <span style={{ fontWeight: 600, color: '#111827' }}>{authorName}</span>
                    {post.author?.role && <span style={{ color: '#9ca3af' }}>{post.author.role}</span>}
                    {post.category && <>
                      <span style={{ color: '#d1d5db' }}>·</span>
                      <span style={{ color: '#6b7280' }}>Asked in</span>
                      <span style={{ color: '#374151', fontWeight: 500 }}>{post.category}</span>
                    </>}
                    <span style={{ marginLeft: 'auto', color: '#9ca3af', fontSize: 11.5 }}>{formatRelativeTime(post.created_at)}</span>
                  </div>

                  {/* Title */}
                  <Link
                    to={`/forum/${post.id}`}
                    style={{ fontSize: 14, fontWeight: 600, color: '#111827', textDecoration: 'none', lineHeight: 1.45, display: 'block', marginBottom: post.body ? 4 : 0, transition: 'color 0.12s' }}
                    onMouseEnter={e => e.currentTarget.style.color = '#6d28d9'}
                    onMouseLeave={e => e.currentTarget.style.color = '#111827'}
                  >
                    {post.title}
                  </Link>

                  {/* Body snippet */}
                  {post.body && (
                    <p style={{ margin: '0 0 8px', fontSize: 13, color: '#6b7280', lineHeight: 1.55, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                      {post.body}
                    </p>
                  )}

                  {/* Stats */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginTop: 8 }}>
                    <button onClick={() => handleVote(post.id, 1)} aria-label="Upvote"
                      style={{ display: 'inline-flex', alignItems: 'center', gap: 3, background: 'none', border: 'none', padding: 0, cursor: 'pointer', fontFamily: 'inherit', fontSize: 12.5, fontWeight: 500, color: voted === 1 ? '#6d28d9' : '#9ca3af', transition: 'color 0.12s' }}
                      onMouseEnter={e => e.currentTarget.style.color = '#6d28d9'}
                      onMouseLeave={e => e.currentTarget.style.color = voted === 1 ? '#6d28d9' : '#9ca3af'}
                    >
                      <ChevronUp size={14} />{post.upvotes}
                    </button>

                    <button onClick={() => handleVote(post.id, -1)} aria-label="Downvote"
                      style={{ display: 'inline-flex', alignItems: 'center', gap: 3, background: 'none', border: 'none', padding: 0, cursor: 'pointer', fontFamily: 'inherit', fontSize: 12.5, fontWeight: 500, color: voted === -1 ? '#dc2626' : '#9ca3af', transition: 'color 0.12s' }}
                      onMouseEnter={e => e.currentTarget.style.color = '#dc2626'}
                      onMouseLeave={e => e.currentTarget.style.color = voted === -1 ? '#dc2626' : '#9ca3af'}
                    >
                      <ChevronDown size={14} />{post.downvotes}
                    </button>

                    {post.view_count != null && (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3, fontSize: 12.5, color: '#9ca3af' }}>
                        <Eye size={13} />{post.view_count}
                      </span>
                    )}

                    <Link to={`/forum/${post.id}`}
                      style={{ display: 'inline-flex', alignItems: 'center', gap: 3, fontSize: 12.5, color: '#9ca3af', textDecoration: 'none', transition: 'color 0.12s' }}
                      onMouseEnter={e => e.currentTarget.style.color = '#374151'}
                      onMouseLeave={e => e.currentTarget.style.color = '#9ca3af'}
                    >
                      <MessageSquare size={13} />{post.answer_count}
                    </Link>
                  </div>
                </div>
              </article>
            )
          })}
        </section>

        {/* Sidebar */}
        <aside style={{ paddingLeft: 24, paddingTop: 15, position: 'sticky', top: 52, alignSelf: 'start', height: 'calc(100vh - 52px)', overflowY: 'auto' }}>
          <p style={{ margin: '0 0 14px', fontSize: 13, fontWeight: 600, color: '#374151' }}>New thread</p>

          <form onSubmit={handleCreatePost} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <input
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="Title"
              required
              style={{ height: 36, borderRadius: 6, border: '1px solid #e5e7eb', padding: '0 10px', fontSize: 13, color: '#111827', outline: 'none', fontFamily: 'inherit', background: '#f9fafb', transition: 'border-color 0.15s, box-shadow 0.15s', width: '100%', boxSizing: 'border-box' }}
              onFocus={e => { e.currentTarget.style.borderColor = '#6d28d9'; e.currentTarget.style.boxShadow = '0 0 0 2px rgba(109,40,217,0.1)' }}
              onBlur={e => { e.currentTarget.style.borderColor = '#e5e7eb'; e.currentTarget.style.boxShadow = 'none' }}
            />

            <textarea
              value={detail}
              onChange={e => setDetail(e.target.value)}
              rows={6}
              placeholder="What's your question? Add context…"
              style={{ borderRadius: 6, border: '1px solid #e5e7eb', padding: '9px 10px', fontSize: 13, color: '#111827', outline: 'none', fontFamily: 'inherit', background: '#f9fafb', resize: 'none', lineHeight: 1.55, transition: 'border-color 0.15s, box-shadow 0.15s', width: '100%', boxSizing: 'border-box' }}
              onFocus={e => { e.currentTarget.style.borderColor = '#6d28d9'; e.currentTarget.style.boxShadow = '0 0 0 2px rgba(109,40,217,0.1)' }}
              onBlur={e => { e.currentTarget.style.borderColor = '#e5e7eb'; e.currentTarget.style.boxShadow = 'none' }}
            />

            {postError && (
              <p style={{ margin: 0, fontSize: 12, color: '#dc2626', background: '#fef2f2', borderRadius: 5, padding: '7px 10px' }}>{postError}</p>
            )}

            <button
              type="submit"
              disabled={posting}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, padding: '9px 0', borderRadius: 6, background: posting ? '#7c3aed' : '#6d28d9', color: '#fff', fontSize: 13, fontWeight: 600, border: 'none', cursor: posting ? 'not-allowed' : 'pointer', fontFamily: 'inherit', transition: 'background 0.15s', width: '100%' }}
              onMouseEnter={e => { if (!posting) e.currentTarget.style.background = '#5b21b6' }}
              onMouseLeave={e => { if (!posting) e.currentTarget.style.background = '#6d28d9' }}
            >
              <Send size={13} />
              {posting ? 'Posting…' : 'Post'}
            </button>
          </form>

          <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid #f3f4f6' }}>
            <p style={{ margin: '0 0 8px', fontSize: 11, fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Tips</p>
            {['Be specific', 'Add context', 'Search first', 'Stay on topic'].map(tip => (
              <p key={tip} style={{ margin: '0 0 5px', fontSize: 12, color: '#6b7280', lineHeight: 1.4 }}>· {tip}</p>
            ))}
          </div>
        </aside>
      </div>
    </div>
  )
}