"""Structural pass: partition document blocks into heading-scoped sections (PRD §4.2, §4.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from akl.ingestion.models import Block, HeadingBlock, PageBreakBlock, UnifiedDocument


@dataclass
class Section:
    heading_path: tuple[str, ...]
    heading_level: int | None
    blocks: list[Block] = field(default_factory=list)

    @property
    def start_char(self) -> int:
        return min((b.start_char for b in self.blocks), default=0)

    @property
    def end_char(self) -> int:
        return max((b.end_char for b in self.blocks), default=0)


def build_sections(doc: UnifiedDocument) -> list[Section]:
    """Walk blocks in order; each heading starts a new section carrying the heading stack.

    Headings with no body are folded into the next section's ``heading_path`` (no
    ``heading_only`` chunk is emitted — PRD §4.3). Page breaks are skipped (page
    numbers come from ``doc.page_map``).
    """
    sections: list[Section] = []
    stack: list[HeadingBlock] = []
    current: Section | None = None

    for block in doc.blocks:
        if isinstance(block, PageBreakBlock):
            continue
        if isinstance(block, HeadingBlock):
            while stack and stack[-1].level >= block.level:
                stack.pop()
            stack.append(block)
            if current is not None and current.blocks:
                sections.append(current)
            current = Section(tuple(h.text for h in stack), block.level)
            continue
        if current is None:
            current = Section((), None)
        current.blocks.append(block)

    if current is not None and current.blocks:
        sections.append(current)
    return sections
