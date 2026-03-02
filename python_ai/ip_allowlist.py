"""
IP Allow-Listing Middleware for NEO Hybrid AI.

Restricts access to configured endpoints based on
client IP address.  Supports individual addresses
and CIDR subnets.  Health and readiness endpoints
are exempted by default.
"""

import ipaddress
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

logger = logging.getLogger(__name__)

# Paths that are always accessible regardless of IP.
DEFAULT_BYPASS: Set[str] = {
    "/health",
    "/healthz",
    "/ready",
    "/readiness",
    "/metrics",
    "/metrics/prometheus",
}


def _parse_networks(
    entries: Sequence[str],
) -> List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]:
    """Parse CIDR strings into network objects.

    Single addresses (e.g. ``192.168.1.1``) are
    converted to single-host networks automatically.

    Args:
        entries: Sequence of IP or CIDR strings.

    Returns:
        List of parsed network objects.
    """
    nets: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = []
    for entry in entries:
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning("Skipping invalid IP/CIDR: %s", entry)
    return nets


def is_allowed(
    remote_ip: str,
    networks: Sequence[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]],
) -> bool:
    """Check whether *remote_ip* falls within any network.

    Args:
        remote_ip: Client IP address string.
        networks: Parsed network objects to match against.

    Returns:
        ``True`` if the IP is within at least one network.
    """
    try:
        addr = ipaddress.ip_address(remote_ip)
    except ValueError:
        return False
    return any(addr in net for net in networks)


class IPAllowListMiddleware:
    """ASGI middleware that rejects requests from IPs not in the allow-list.

    Requests to *bypass_paths* are always permitted.
    When the allow-list is empty **all** requests are
    permitted (open mode).

    Args:
        app: Inner ASGI application.
        allowed: IP addresses or CIDR subnets.
        bypass_paths: Paths exempt from filtering.
        reject_status: HTTP status code for blocked
            requests (default ``403``).
    """

    def __init__(
        self,
        app: Any,
        allowed: Optional[Sequence[str]] = None,
        bypass_paths: Optional[Set[str]] = None,
        reject_status: int = 403,
    ) -> None:
        """Initialise the middleware."""
        self.app = app
        self._networks = _parse_networks(allowed) if allowed else []
        self._bypass = (
            bypass_paths if bypass_paths is not None else DEFAULT_BYPASS
        )
        self._status = reject_status
        self._stats: Dict[str, int] = {
            "allowed": 0,
            "blocked": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        """Return allow/block counters (read-only copy)."""
        return dict(self._stats)

    def update_allowlist(self, entries: Sequence[str]) -> None:
        """Replace the current allow-list at runtime.

        Args:
            entries: New IP/CIDR entries.
        """
        self._networks = _parse_networks(entries)
        logger.info(
            "Allow-list updated: %d network(s)",
            len(self._networks),
        )

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        """Handle an ASGI connection."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Bypass paths are always allowed.
        path: str = scope.get("path", "/")
        if path in self._bypass:
            self._stats["allowed"] += 1
            await self.app(scope, receive, send)
            return

        # Open mode: empty allow-list permits all.
        if not self._networks:
            self._stats["allowed"] += 1
            await self.app(scope, receive, send)
            return

        # Extract client IP (first element of client tuple).
        client = scope.get("client")
        remote_ip = client[0] if client else ""

        if is_allowed(remote_ip, self._networks):
            self._stats["allowed"] += 1
            await self.app(scope, receive, send)
            return

        # Block the request.
        self._stats["blocked"] += 1
        logger.warning("Blocked request from %s", remote_ip)
        body = b"Forbidden"
        await send(
            {
                "type": "http.response.start",
                "status": self._status,
                "headers": [
                    [b"content-type", b"text/plain"],
                    [
                        b"content-length",
                        str(len(body)).encode(),
                    ],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )
