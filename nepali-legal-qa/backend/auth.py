import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from jose import JWTError, jwt
from pydantic import BaseModel


SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")


class TokenData(BaseModel):
    sub: str
    email: str
    name: str
    picture: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_google_token(token: str) -> TokenData:
    """Verify Google ID token and return token data."""
    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )
        
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Invalid token issuer')
        
        return TokenData(
            sub=idinfo['sub'],
            email=idinfo.get('email', ''),
            name=idinfo.get('name', ''),
            picture=idinfo.get('picture', None)
        )
    except ValueError as e:
        raise ValueError(f"Invalid token: {str(e)}")


def verify_access_token(token: str) -> TokenData:
    """Verify JWT access token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub: str = payload.get("sub")
        email: str = payload.get("email")
        name: str = payload.get("name")
        picture: Optional[str] = payload.get("picture")
        
        if sub is None:
            raise ValueError("Invalid token")
        
        return TokenData(sub=sub, email=email, name=name, picture=picture)
    except JWTError:
        raise ValueError("Invalid token")
