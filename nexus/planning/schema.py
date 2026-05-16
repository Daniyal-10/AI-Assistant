"""
nexus/planning/schema.py
────────────────────────
Typed schemas for the planning layer.
Richer than the raw dict the LLM returns.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TaskComplexity(Enum):
    SIMPLE   = "simple"    # single file, no deps, < 50 lines
    MODERATE = "moderate"  # 2-5 files, maybe 1-2 deps
    COMPLEX  = "complex"   # many files, multiple deps, external APIs


class TaskCategory(Enum):
    SCRIPT        = "script"
    DATA          = "data_processing"
    API           = "api_client"
    WEB           = "web_service"
    UTILITY       = "utility"
    UNKNOWN       = "unknown"


@dataclass
class EnrichedPlan:
    """
    A plan dict from the LLM, enriched with pre-generation analysis.
    All original LLM fields are preserved in raw_plan.
    """
    # Original LLM plan fields (passed through unchanged)
    raw_plan: dict

    # Enrichment fields (set by planner, not LLM)
    complexity:      TaskComplexity = TaskComplexity.SIMPLE
    category:        TaskCategory   = TaskCategory.UNKNOWN
    estimated_files: int            = 1
    has_network:     bool           = False
    has_file_io:     bool           = False
    needs_deps:      bool           = False
    feasibility_ok:  bool           = True
    feasibility_notes: List[str]    = field(default_factory=list)

    # Pass-through helpers so engine can treat this like a dict
    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict, including enrichment fields."""
        res = self.raw_plan.copy()
        res.update({
            "_enrichment": {
                "complexity":      self.complexity.value,
                "category":        self.category.value,
                "estimated_files": self.estimated_files,
                "has_network":     self.has_network,
                "has_file_io":     self.has_file_io,
                "needs_deps":      self.needs_deps,
                "feasibility_ok":  self.feasibility_ok,
                "feasibility_notes": self.feasibility_notes,
            }
        })
        return res

    def get(self, key: str, default=None):
        return self.raw_plan.get(key, default)

    def __getitem__(self, key: str):
        return self.raw_plan[key]

    def __setitem__(self, key: str, value):
        self.raw_plan[key] = value

    def __contains__(self, key: str):
        return key in self.raw_plan

    def keys(self):
        return self.raw_plan.keys()

    def items(self):
        return self.raw_plan.items()
