import httpx
from fastapi import HTTPException, status
from src.api.config import settings


async def verify_google_token(id_token: str) -> dict:
    """
    Verifies a Google ID token against Google's public OAuth 2.0 tokeninfo endpoint.
    Returns verified user claims (email, name, sub).
    """
    if not id_token or id_token == "YOUR_GOOGLE_ID_TOKEN_HERE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or placeholder Google ID token provided.",
        )

    google_verify_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"

    async with httpx.AsyncClient() as client:
        response = await client.get(google_verify_url)

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to verify Google ID token with Google servers.",
        )

    claims = response.json()

    # Optional: If GOOGLE_CLIENT_ID is set, verify audience claim
    if settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_ID != "YOUR_GOOGLE_CLIENT_ID_HERE":
        if claims.get("aud") != settings.GOOGLE_CLIENT_ID:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Google token audience mismatch (token was not issued for this client ID).",
            )

    email = claims.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google ID token does not contain a valid email address.",
        )

    return {
        "email": email,
        "full_name": claims.get("name", email.split("@")[0]),
        "google_sub": claims.get("sub"),
    }
