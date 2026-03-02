"""
Risk Governance Workflow for NEO Hybrid AI.

Implements approval gates for high-risk trades,
configurable risk limits, and a review/audit trail
for governance-sensitive operations.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
)

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of a governance request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class RiskPolicy:
    """A configurable risk policy rule.

    Attributes:
        name: Policy identifier.
        description: Human-readable explanation.
        max_position_pct: Max portfolio percentage for
            a single position.
        max_trade_value: Absolute max trade value.
        require_approval_above: Trade value threshold
            requiring human approval.
        cooldown_seconds: Min time between trades on
            the same asset.
    """

    name: str
    description: str = ""
    max_position_pct: float = 10.0
    max_trade_value: float = 50_000.0
    require_approval_above: float = 10_000.0
    cooldown_seconds: float = 300.0


@dataclass
class GovernanceRequest:
    """A request requiring governance approval.

    Attributes:
        request_id: Unique identifier.
        trade_details: Dict with symbol, side, qty, etc.
        policy_name: Which policy triggered the gate.
        reason: Why approval is needed.
        status: Current approval status.
        submitted_at: Unix timestamp of submission.
        decided_at: Unix timestamp of decision.
        decided_by: Identity of the approver/rejecter.
        notes: Free-text notes.
    """

    request_id: str
    trade_details: Dict[str, Any]
    policy_name: str
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    submitted_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None
    decided_by: Optional[str] = None
    notes: str = ""


class RiskGovernor:
    """Enforces risk policies and manages approvals.

    Evaluates proposed trades against configured
    policies and gates high-risk operations behind
    an approval workflow.

    Args:
        policies: Initial set of risk policies.
        request_ttl: Seconds before a pending request
            auto-expires.
    """

    def __init__(
        self,
        policies: Optional[Sequence[RiskPolicy]] = None,
        request_ttl: float = 3600.0,
    ) -> None:
        """Initialise the governor."""
        self._policies: Dict[str, RiskPolicy] = {}
        for p in policies or []:
            self._policies[p.name] = p
        self._ttl = request_ttl
        self._requests: Dict[str, GovernanceRequest] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._last_trade: Dict[str, float] = {}
        self._stats = {
            "checks": 0,
            "approved": 0,
            "rejected": 0,
            "auto_approved": 0,
        }

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def add_policy(self, policy: RiskPolicy) -> None:
        """Register or update a risk policy.

        Args:
            policy: The policy to add.
        """
        with self._lock:
            self._policies[policy.name] = policy

    def get_policy(self, name: str) -> Optional[RiskPolicy]:
        """Get a policy by name.

        Args:
            name: Policy identifier.

        Returns:
            The policy, or ``None``.
        """
        return self._policies.get(name)

    # ------------------------------------------------------------------
    # Trade evaluation
    # ------------------------------------------------------------------

    def evaluate_trade(
        self,
        trade: Dict[str, Any],
        policy_name: str = "default",
        portfolio_value: float = 100_000.0,
    ) -> Dict[str, Any]:
        """Evaluate a proposed trade against policy.

        Args:
            trade: Must contain ``value`` (float) and
                ``symbol`` (str) keys.
            policy_name: Which policy to use.
            portfolio_value: Current portfolio value.

        Returns:
            Dict with ``allowed`` (bool),
            ``requires_approval`` (bool),
            ``violations`` (list), ``request_id``
            (if approval needed).
        """
        self._stats["checks"] += 1
        policy = self._policies.get(policy_name)
        if policy is None:
            # No policy → auto-approve
            self._stats["auto_approved"] += 1
            return {
                "allowed": True,
                "requires_approval": False,
                "violations": [],
            }

        value = float(trade.get("value", 0))
        symbol = str(trade.get("symbol", ""))
        violations: List[str] = []

        # Position size check.
        if portfolio_value > 0:
            pct = value / portfolio_value * 100
            if pct > policy.max_position_pct:
                violations.append(
                    f"Position {pct:.1f}% exceeds "
                    f"max {policy.max_position_pct}%"
                )

        # Absolute value check.
        if value > policy.max_trade_value:
            violations.append(
                f"Value {value} exceeds max " f"{policy.max_trade_value}"
            )

        # Cooldown check.
        now = time.time()
        last = self._last_trade.get(symbol, 0)
        if (now - last) < policy.cooldown_seconds:
            remaining = policy.cooldown_seconds - (now - last)
            violations.append(f"Cooldown: {remaining:.0f}s remaining")

        if violations:
            self._stats["rejected"] += 1
            return {
                "allowed": False,
                "requires_approval": False,
                "violations": violations,
            }

        # Approval gate.
        if value > policy.require_approval_above:
            req = self._create_request(
                trade, policy_name, "Value exceeds gate"
            )
            return {
                "allowed": False,
                "requires_approval": True,
                "violations": [],
                "request_id": req.request_id,
            }

        # Auto-approve.
        self._stats["auto_approved"] += 1
        self._last_trade[symbol] = now
        return {
            "allowed": True,
            "requires_approval": False,
            "violations": [],
        }

    # ------------------------------------------------------------------
    # Approval workflow
    # ------------------------------------------------------------------

    def _create_request(
        self,
        trade: Dict[str, Any],
        policy_name: str,
        reason: str,
    ) -> GovernanceRequest:
        """Create a pending governance request.

        Args:
            trade: Trade details dict.
            policy_name: Triggering policy.
            reason: Why approval is needed.

        Returns:
            New :class:`GovernanceRequest`.
        """
        with self._lock:
            self._seq += 1
            rid = f"GOV-{self._seq:06d}"
            req = GovernanceRequest(
                request_id=rid,
                trade_details=trade,
                policy_name=policy_name,
                reason=reason,
            )
            self._requests[rid] = req
        logger.info(
            "Governance request %s created: %s",
            rid,
            reason,
        )
        return req

    def approve(
        self,
        request_id: str,
        approver: str,
        notes: str = "",
    ) -> bool:
        """Approve a pending request.

        Args:
            request_id: The request to approve.
            approver: Identity of the approver.
            notes: Optional notes.

        Returns:
            ``True`` if approved, ``False`` if not found
            or not pending.
        """
        with self._lock:
            req = self._requests.get(request_id)
            if not req:
                return False
            if req.status != ApprovalStatus.PENDING:
                return False
            req.status = ApprovalStatus.APPROVED
            req.decided_at = time.time()
            req.decided_by = approver
            req.notes = notes
            self._stats["approved"] += 1

            symbol = req.trade_details.get("symbol", "")
            if symbol:
                self._last_trade[symbol] = time.time()
        logger.info(
            "Request %s approved by %s",
            request_id,
            approver,
        )
        return True

    def reject(
        self,
        request_id: str,
        rejector: str,
        notes: str = "",
    ) -> bool:
        """Reject a pending request.

        Args:
            request_id: The request to reject.
            rejector: Identity of the rejector.
            notes: Optional notes.

        Returns:
            ``True`` if rejected successfully.
        """
        with self._lock:
            req = self._requests.get(request_id)
            if not req:
                return False
            if req.status != ApprovalStatus.PENDING:
                return False
            req.status = ApprovalStatus.REJECTED
            req.decided_at = time.time()
            req.decided_by = rejector
            req.notes = notes
            self._stats["rejected"] += 1
        logger.info(
            "Request %s rejected by %s",
            request_id,
            rejector,
        )
        return True

    def expire_stale(self) -> int:
        """Expire pending requests older than TTL.

        Returns:
            Number of requests expired.
        """
        now = time.time()
        expired = 0
        with self._lock:
            for req in self._requests.values():
                if (
                    req.status == ApprovalStatus.PENDING
                    and (now - req.submitted_at) > self._ttl
                ):
                    req.status = ApprovalStatus.EXPIRED
                    req.decided_at = now
                    expired += 1
        return expired

    def get_pending(self) -> List[GovernanceRequest]:
        """Return all pending requests.

        Returns:
            List of :class:`GovernanceRequest`.
        """
        with self._lock:
            return [
                r
                for r in self._requests.values()
                if r.status == ApprovalStatus.PENDING
            ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return governance summary.

        Returns:
            Dict with policy count, request stats.
        """
        with self._lock:
            pending = sum(
                1
                for r in self._requests.values()
                if r.status == ApprovalStatus.PENDING
            )
        return {
            "policies": len(self._policies),
            "pending_requests": pending,
            "total_requests": len(self._requests),
            **self._stats,
        }
