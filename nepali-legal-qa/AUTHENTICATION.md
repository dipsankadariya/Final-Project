# Google Authentication Setup Guide

## Overview
This project now includes Google OAuth2 authentication for user login.

## Backend Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

The following packages have been added for authentication:
- `google-auth-oauthlib>=1.2.0`
- `google-auth>=2.28.0`
- `python-jose[cryptography]>=3.3.0`
- `PyJWT>=2.8.1`

### 2. Environment Variables
Create or update your `.env` file in the `backend/` directory:

```env
# Google OAuth2 Client ID from Google Cloud Console
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com

# Secret key for JWT token signing (change this in production!)
SECRET_KEY=your-secret-key-change-this-in-production

# Existing variables
MODEL_ID=zeri000/nepali_legal_qwen_merged_4
DOC_FILE_PATH=../../../augmented_nepali_legal_rag.txt
GROQ_API_KEY=your-groq-api-key
```

### 3. Authentication Endpoints
- `POST /api/auth/google` - Login with Google token
- `GET /api/auth/verify` - Verify and get current user info
- `POST /api/query` - Query endpoint (now accepts optional Authorization header)

## Frontend Setup

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Environment Variables
Create `.env.local` in the `frontend/` directory:

```env
VITE_GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
# Optional: set this if backend is on a different domain
# VITE_API_BASE=http://localhost:8000
```

### 3. Components Added
- `src/GoogleLogin.jsx` - Google login button and login card component

## Google Cloud Setup

### 1. Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the "Google+ API"

### 2. Create OAuth2 Credentials
1. Go to "Credentials" in the left menu
2. Click "Create Credentials" → "OAuth 2.0 Client ID"
3. Choose "Web application"
4. Add authorized JavaScript origins:
   - `http://localhost:3000` (for local development)
   - `http://localhost:8000` (if frontend is on port 3000)
   - Add your production domain
5. Add authorized redirect URIs:
   - `http://localhost:3000` (for local development)
   - Add your production domain
6. Copy the Client ID and use it in both backend and frontend

## How Authentication Works

### Login Flow
1. User clicks "Sign in with Google" button
2. Google authentication popup opens
3. User authenticates with their Google account
4. Browser receives an ID token from Google
5. Frontend sends ID token to backend `/api/auth/google` endpoint
6. Backend verifies the token and creates a JWT access token
7. Frontend stores the JWT in localStorage
8. User is logged in and can access the app

### Protected API Calls
- The frontend automatically includes the JWT token in the `Authorization` header for all API requests
- Backend validates the token before processing requests
- If token is invalid or expired, API returns 401 Unauthorized

## Security Notes

⚠️ **Important for Production:**
1. Change the `SECRET_KEY` in your backend `.env` to a secure random string
2. Keep `GOOGLE_CLIENT_ID` and other credentials secret
3. Use HTTPS in production
4. Set secure cookie flags if using session cookies
5. Consider adding token refresh logic for long-lived sessions

## Testing Authentication

### Test Backend Authentication
```bash
# Start backend
python -m uvicorn main:app --reload

# In another terminal, test the health endpoint
curl http://localhost:8000/api/health

# Test the verify endpoint (this will fail without a valid token)
curl -H "Authorization: Bearer invalid-token" http://localhost:8000/api/auth/verify
```

### Test Frontend
1. Start frontend: `npm run dev`
2. You should see the login screen
3. Click "Sign in with Google"
4. Complete Google authentication
5. You should be redirected to the main app

## Existing Code Preservation

✅ All existing code has been preserved:
- The original `/api/query` endpoint still works without authentication
- The `queryLegal()` function automatically adds auth token if available
- All existing UI components and functionality remain unchanged
- User profile component is added to the header (only visible when logged in)
