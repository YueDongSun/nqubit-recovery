"""Physics Hybrid V10 variable-N research models.

V10 is research-only.  This package has no hardware-write authority and deliberately
does not import the experiment runner, AWG uploader, FPGA programmer, or delta manager.
"""

from .contracts import V10ResearchAuthority, V10ShotRecord

__all__ = ["V10ResearchAuthority", "V10ShotRecord"]
__version__ = "10.0.0-research"
