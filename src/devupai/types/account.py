"""Types of the account endpoints: ``GET /v1/user/balance``, ``GET /v1/me`` and ``GET /health``."""

from typing import Literal

from devupai.types._base import DevupModel

__all__ = [
    "AccountAvatar",
    "AccountProfile",
    "BalanceResponse",
    "HealthResponse",
]


class BalanceResponse(DevupModel):
    """The response of ``GET /v1/user/balance``."""

    balance_dzd: float
    """The account balance, in Algerian dinars (DZD)."""


class AccountAvatar(DevupModel):
    """Where to fetch the account's profile picture, and its current version."""

    url: str
    """The absolute URL of ``GET /v1/me/avatar``."""
    etag: str
    """The entity tag of the current picture, quotes included; it changes when the picture changes."""


class AccountProfile(DevupModel):
    """The response of ``GET /v1/me``: the profile of the account that owns the API key."""

    object: Literal["account_profile"] | str
    display_name: str | None
    """The account's name, or ``None`` when the account has no name."""
    initial: str
    """One uppercase letter or digit for the account."""
    email_hint: str | None
    """A masked form of the account's email, or ``None`` when the email is not known."""
    avatar: AccountAvatar | None
    """The profile picture, or ``None`` when the account has none."""


class HealthResponse(DevupModel):
    """The response of ``GET /health``."""

    status: Literal["ok", "error"] | str
