"""
NEXUS Event Model & EventBus
────────────────────────────
This module defines the typed event contracts and synchronous EventBus
for the task, stage, and repair lifecycles.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Type, TypeVar
import uuid

# ── Base Event Contract ───────────────────────────────────────────────────────

@dataclass
class NexusEvent:
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the event to a dictionary for logging/boundaries/export layers.
        Converts datetime timestamps to string representation only at this layer.
        """
        res = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "event_type": self.__class__.__name__,
        }
        # Safely copy dynamic subclass-specific fields
        for k, v in self.__dict__.items():
            if k not in ("event_id", "timestamp", "metadata"):
                res[k] = v
        return res


# ── Task Lifecycle Events ────────────────────────────────────────────────────

@dataclass
class TaskStartedEvent(NexusEvent):
    task_id: str = ""
    raw_input: str = ""


@dataclass
class TaskFinishedEvent(NexusEvent):
    task_id: str = ""
    success: bool = False
    summary: str = ""
    status: str = ""


# ── Pipeline Stage Lifecycle Events ──────────────────────────────────────────

@dataclass
class PipelineStageStartedEvent(NexusEvent):
    task_id: str = ""
    stage: str = ""


@dataclass
class PipelineStageFinishedEvent(NexusEvent):
    task_id: str = ""
    stage: str = ""
    status: str = ""


# ── Repair Lifecycle Events ──────────────────────────────────────────────────

@dataclass
class RepairLoopStartedEvent(NexusEvent):
    task_id: str = ""
    max_iterations: int = 0


@dataclass
class RepairIterationStartedEvent(NexusEvent):
    task_id: str = ""
    iteration: int = 0
    error_category: str = ""


@dataclass
class RepairIterationFinishedEvent(NexusEvent):
    task_id: str = ""
    iteration: int = 0
    success: bool = False
    error_category: str = ""


@dataclass
class RepairLoopFinishedEvent(NexusEvent):
    task_id: str = ""
    success: bool = False
    iterations_used: int = 0


# ── EventBus Implementation ──────────────────────────────────────────────────

T = TypeVar("T", bound=NexusEvent)
EventHandler = Callable[[T], None]


class EventBus:
    """
    Instance-scoped, synchronous, fail-safe event broker.
    Ensures absolute isolation for multi-session and async execution models.
    """
    def __init__(self) -> None:
        self._subscribers: Dict[Type[NexusEvent], List[EventHandler]] = {}

    def subscribe(self, event_type: Type[T], handler: EventHandler[T]) -> None:
        """Subscribe a callback handler to a specific event type or base type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[T], handler: EventHandler[T]) -> None:
        """Unsubscribe a callback handler from a specific event type or base type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass

    def emit(self, event: NexusEvent) -> None:
        """
        Synchronously publish an event to all matching handlers.
        Wraps executions to protect the caller from subscriber failures.
        """
        for event_type, handlers in self._subscribers.items():
            if isinstance(event, event_type):
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception:
                        # Fail-safe design: subscriber failures never crash execution
                        pass
