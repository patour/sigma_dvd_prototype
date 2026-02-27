"""
Current source waveform classes and parsing helpers.

Contains: ContextVars for wscale/dc_only, InstanceInfo, Pulse, PWL,
CurrentSource, _DCOnlyCurrentSource, parsing functions (_parse_pulse,
_parse_pwl, _parse_current_source_line), PreparedSource, _prepare_instance_source.

Imports from parser.spice_lexer for _parse_spice_value and regex patterns.
"""

import bisect
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

from .spice_lexer import (
    _parse_spice_value,
    _RE_PULSE, _RE_PWL,
    I_TO_MA,
)


# =============================================================================
# Thread-safe ContextVars
# =============================================================================

# Wscale control: whether waveform scaling factors are applied
_apply_wscale: ContextVar[bool] = ContextVar('apply_wscale', default=True)


def get_apply_wscale() -> bool:
    """Get current wscale application setting (thread-safe).

    Returns:
        True if wscale factors should be applied to pulse/PWL evaluations,
        False otherwise.
    """
    return _apply_wscale.get()


def set_apply_wscale(enabled: bool) -> None:
    """Set wscale application for current context (thread-safe).

    Each thread/async context has its own independent value. Setting this
    in one thread does not affect other threads.

    Args:
        enabled: If True, wscale factors are applied to pulse/PWL evaluations.
                 If False, wscale is parsed but not applied.
    """
    _apply_wscale.set(enabled)


# DC-only optimization control
_optimize_dc_only: ContextVar[bool] = ContextVar('_optimize_dc_only', default=True)


def get_optimize_dc_only() -> bool:
    """Return whether DC-only current sources use the lightweight representation.

    Returns:
        True if DC-only sources use _DCOnlyCurrentSource (default),
        False if all sources use full CurrentSource dataclass.
    """
    return _optimize_dc_only.get()


def set_optimize_dc_only(value: bool) -> None:
    """Enable/disable lightweight DC-only current source optimization.

    When enabled (default), current sources with only a DC value (no pulse/PWL
    waveforms) use _DCOnlyCurrentSource with 4 slots instead of CurrentSource
    with 9+ attributes, saving ~400 bytes per instance.

    Args:
        value: If True, use lightweight representation for DC-only sources.
               If False, use full CurrentSource for all sources.
    """
    _optimize_dc_only.set(value)


# =============================================================================
# Instance Info
# =============================================================================

@dataclass(slots=True)
class InstanceInfo:
    """
    Parsed instance name information.

    Instance name format:
      i_<instance_name>:<vdd_net>:<vdd_pin>:<vss_net>:<vss_pin>:<tile_x>:<tile_y>[:<extra>]

    Example:
      i_U123/cell:VDD_XLV:VDD:0:0:5:3:0

    Note: instance_name is a computed property derived from full_name to save memory.
    """
    full_name: str
    vdd_net: Optional[str] = None
    vdd_pin: Optional[str] = None
    vss_net: Optional[str] = None
    vss_pin: Optional[str] = None
    tile_x: int = 0
    tile_y: int = 0

    @property
    def instance_name(self) -> str:
        """Derived from full_name by removing 'i_' prefix and splitting on ':'."""
        name = self.full_name
        if name.lower().startswith('i_'):
            name = name[2:]
        return name.split(':')[0]

    @classmethod
    def parse(cls, name: str, delimiter: str = ':') -> 'InstanceInfo':
        """Parse instance name to extract net and location info."""
        info = cls(full_name=name)

        # Remove 'i_' or 'I_' prefix for parsing
        parse_name = name
        if parse_name.lower().startswith('i_'):
            parse_name = parse_name[2:]

        parts = parse_name.split(delimiter)

        # instance_name is now a computed property (parts[0])
        if len(parts) >= 2:
            info.vdd_net = parts[1] if parts[1] != '0' else None
        if len(parts) >= 3:
            info.vdd_pin = parts[2] if parts[2] != '0' else None
        if len(parts) >= 4:
            info.vss_net = parts[3] if parts[3] != '0' else None
        if len(parts) >= 5:
            info.vss_pin = parts[4] if parts[4] != '0' else None
        if len(parts) >= 6:
            try:
                info.tile_x = int(parts[5])
            except ValueError:
                pass
        if len(parts) >= 7:
            try:
                info.tile_y = int(parts[6])
            except ValueError:
                pass

        return info

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON-compatible storage.

        Note: instance_name is included for backward compatibility but
        will be recomputed from full_name when deserializing.
        """
        return {
            'full_name': self.full_name,
            'instance_name': self.instance_name,  # Computed property, for backward compat
            'vdd_net': self.vdd_net,
            'vdd_pin': self.vdd_pin,
            'vss_net': self.vss_net,
            'vss_pin': self.vss_pin,
            'tile_x': self.tile_x,
            'tile_y': self.tile_y
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'InstanceInfo':
        """Reconstruct from dictionary.

        Note: instance_name from dict is ignored - it will be recomputed
        from full_name via the property.
        """
        return cls(
            full_name=d.get('full_name', ''),
            # instance_name is now a computed property, not stored
            vdd_net=d.get('vdd_net'),
            vdd_pin=d.get('vdd_pin'),
            vss_net=d.get('vss_net'),
            vss_pin=d.get('vss_pin'),
            tile_x=d.get('tile_x', 0),
            tile_y=d.get('tile_y', 0)
        )


# =============================================================================
# Pulse Waveform
# =============================================================================

@dataclass(slots=True)
class Pulse:
    """
    Pulse waveform definition.

    All values in base units (Amperes for current). Convert to mA when storing.
    """
    v1: float = 0.0       # Initial value
    v2: float = 0.0       # Pulsed value
    delay: float = 0.0    # Delay time (s)
    rt: float = 0.0       # Rise time (s)
    ft: float = 0.0       # Fall time (s)
    width: float = 0.0    # Pulse width (s)
    period: float = 0.0   # Period (s), 0 = non-periodic

    def evaluate(self, time: float) -> float:
        """Evaluate pulse value at given time (standard SPICE timing).

        Pulse timing interpretation (matches C++ SimPWL):
        - delay = START time (when signal begins rising from v1)
        - rt = rise time from v1 to v2
        - width = duration at v2 after rise completes
        - ft = fall time from v2 back to v1

        Timeline: [delay] --rise--> [delay+rt] --high--> [delay+rt+width] --fall--> [delay+rt+width+ft]
        """
        # Step 1: Calculate time relative to pulse START (delay)
        t_rel = time - self.delay

        # Step 2: Apply periodic wrapping
        if self.period > 0:
            # Python's % handles negatives correctly: -2 % 10 = 8
            t_rel = t_rel % self.period
        elif t_rel < 0:
            # Non-periodic, before pulse start
            return self.v1

        # Step 3: Check if beyond pulse envelope within one period
        # Pulse duration = rt + width + ft (from start to end)
        pulse_duration = self.rt + self.width + self.ft
        if t_rel >= pulse_duration:
            return self.v1

        # Step 4: Evaluate phase within pulse envelope
        if t_rel < self.rt:
            # Rise phase: [0, rt)
            if self.rt > 0:
                return self.v1 + (self.v2 - self.v1) * (t_rel / self.rt)
            return self.v2
        elif t_rel < self.rt + self.width:
            # High phase: [rt, rt + width)
            return self.v2
        else:
            # Fall phase: [rt + width, rt + width + ft)
            if self.ft > 0:
                t_fall = t_rel - self.rt - self.width
                return self.v2 + (self.v1 - self.v2) * (t_fall / self.ft)
            return self.v1

    def get_dc(self) -> float:
        """Calculate average DC value of pulse over one period."""
        if self.period <= 0:
            return 0.0
        rise_area = 0.5 * self.rt * (self.v1 + self.v2)
        high_area = self.width * self.v2
        fall_area = 0.5 * self.ft * (self.v1 + self.v2)
        low_time = self.period - self.rt - self.width - self.ft
        low_area = max(0, low_time) * self.v1
        total_area = rise_area + high_area + fall_area + low_area
        return total_area / self.period

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'v1': self.v1, 'v2': self.v2, 'delay': self.delay,
            'rt': self.rt, 'ft': self.ft, 'width': self.width, 'period': self.period
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Pulse':
        """Reconstruct from dictionary."""
        return cls(
            v1=d.get('v1', 0.0), v2=d.get('v2', 0.0), delay=d.get('delay', 0.0),
            rt=d.get('rt', 0.0), ft=d.get('ft', 0.0), width=d.get('width', 0.0),
            period=d.get('period', 0.0)
        )


# =============================================================================
# PWL Waveform
# =============================================================================

@dataclass(slots=True)
class PWL:
    """
    Piece-wise linear waveform.

    points: List of (time, value) tuples.
    """
    points: List[Tuple[float, float]] = field(default_factory=list)
    period: float = 0.0
    delay: float = 0.0
    _times_cache: Optional[Tuple[float, ...]] = field(default=None, init=False, repr=False)

    def _get_times(self) -> Tuple[float, ...]:
        """Get cached tuple of time values for binary search."""
        if self._times_cache is None:
            self._times_cache = tuple(p[0] for p in self.points)
        return self._times_cache

    def evaluate(self, time: float) -> float:
        """Evaluate PWL value at given time using binary search (O(log N))."""
        if not self.points:
            return 0.0

        t = time - self.delay
        if self.period > 0 and t >= 0:
            t = t % self.period

        if t <= self.points[0][0]:
            return self.points[0][1]

        if t >= self.points[-1][0]:
            if self.period > 0:
                return self.points[0][1]
            return self.points[-1][1]

        # Binary search for the interval containing t
        times = self._get_times()
        i = bisect.bisect_right(times, t) - 1

        t1, v1 = self.points[i]
        t2, v2 = self.points[i + 1]
        if t2 == t1:
            return v1
        return v1 + (v2 - v1) * (t - t1) / (t2 - t1)

    def get_dc(self) -> float:
        """Calculate average DC value of PWL over one period."""
        if not self.points or len(self.points) < 2:
            if self.points:
                return self.points[0][1]
            return 0.0

        if self.period <= 0:
            return 0.0

        total_area = 0.0
        for i in range(len(self.points) - 1):
            t1, v1 = self.points[i]
            t2, v2 = self.points[i + 1]
            total_area += 0.5 * (v1 + v2) * (t2 - t1)

        if self.period > self.points[-1][0]:
            t_last, v_last = self.points[-1]
            t_first, v_first = self.points[0]
            remaining = self.period - t_last + t_first
            total_area += 0.5 * (v_last + v_first) * remaining

        return total_area / self.period

    def __reduce__(self):
        """Custom pickle support - only serialize init fields, not _times_cache."""
        return (PWL, (self.points, self.period, self.delay))

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {'points': self.points, 'period': self.period, 'delay': self.delay}

    @classmethod
    def from_dict(cls, d: Dict) -> 'PWL':
        """Reconstruct from dictionary."""
        return cls(
            points=[(p[0], p[1]) for p in d.get('points', [])],
            period=d.get('period', 0.0),
            delay=d.get('delay', 0.0)
        )


# =============================================================================
# CurrentSource
# =============================================================================

@dataclass(slots=True)
class CurrentSource:
    """
    Current source instance with full waveform data.

    All current values are stored in mA for consistency with the PDN parser.
    Use get_static_current() for DC analysis, get_current_at_time(t) for transient.

    The wscale parameter scales pulse/PWL waveform values (but NOT dc_value).
    This matches C++ SimPWL behavior where wscale is applied to waveform evaluations.
    """
    name: str
    node1: str
    node2: str
    dc_value: float = 0.0                           # DC component (mA)
    static_value: Optional[float] = None            # Static value override (mA)
    pulses: List[Pulse] = field(default_factory=list)
    pwls: List[PWL] = field(default_factory=list)
    info: Optional[InstanceInfo] = None
    wscale: float = 1.0                             # Waveform scaling factor

    def has_waveform_data(self) -> bool:
        """Check if instance has any dynamic waveform data (Pulse or PWL)."""
        return len(self.pulses) > 0 or len(self.pwls) > 0

    def has_current_data(self) -> bool:
        """Check if instance has any current data."""
        return (self.dc_value != 0.0 or
                self.static_value is not None or
                self.has_waveform_data())

    def get_static_current(self) -> float:
        """Get static/DC current value (mA).

        If static_value is set, returns dc_value + static_value (wscale not applied).
        Otherwise, computes DC average from waveforms with wscale applied (if enabled).
        """
        if self.static_value is not None:
            return self.dc_value + self.static_value

        # Apply wscale to waveform DC averages (not to dc_value)
        scale = self.wscale if get_apply_wscale() else 1.0

        total = self.dc_value  # DC NOT scaled (matches C++ behavior)
        for pulse in self.pulses:
            total += pulse.get_dc() * scale
        for pwl in self.pwls:
            total += pwl.get_dc() * scale
        return total

    def get_current_at_time(self, time: float) -> float:
        """Get current value at specified time (mA).

        Applies wscale to pulse/PWL waveform evaluations (if enabled).
        DC component is NOT scaled (matches C++ SimPWL behavior).
        """
        scale = self.wscale if get_apply_wscale() else 1.0

        total = self.dc_value  # DC NOT scaled
        for pulse in self.pulses:
            total += pulse.evaluate(time) * scale
        for pwl in self.pwls:
            total += pwl.evaluate(time) * scale
        return total

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON-compatible storage."""
        return {
            'name': self.name,
            'node1': self.node1,
            'node2': self.node2,
            'dc_value': self.dc_value,
            'static_value': self.static_value,
            'pulses': [p.to_dict() for p in self.pulses],
            'pwls': [p.to_dict() for p in self.pwls],
            'info': self.info.to_dict() if self.info else None,
            'wscale': self.wscale,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'CurrentSource':
        """Reconstruct from dictionary."""
        return cls(
            name=d.get('name', ''),
            node1=d.get('node1', ''),
            node2=d.get('node2', ''),
            dc_value=d.get('dc_value', 0.0),
            static_value=d.get('static_value'),
            pulses=[Pulse.from_dict(p) for p in d.get('pulses', [])],
            pwls=[PWL.from_dict(p) for p in d.get('pwls', [])],
            info=InstanceInfo.from_dict(d['info']) if d.get('info') else None,
            wscale=d.get('wscale', 1.0),  # Default 1.0 for backward compat
        )


# =============================================================================
# _DCOnlyCurrentSource (lightweight)
# =============================================================================

class _DCOnlyCurrentSource:
    """Lightweight stand-in for CurrentSource when only a DC value exists.

    Uses plain __slots__ (4 attrs) instead of @dataclass(slots=True) with 9+
    attributes. Provides the same duck-typed interface consumed by
    vectorized_sources, transient_solver, and adjoint_sensitivity.

    Note: wscale is NOT stored because it only applies to waveforms, and
    DC-only sources have no waveforms. The wscale property returns 1.0.

    Memory savings: ~400 bytes per instance vs full CurrentSource.
    """
    __slots__ = ('node1', 'node2', 'dc_value', 'info')

    def __init__(self, node1: str, node2: str, dc_value: float,
                 info: Optional[InstanceInfo] = None):
        self.node1 = node1
        self.node2 = node2
        self.dc_value = dc_value
        self.info = info

    # --- Duck-typed interface expected by consumers ---

    @property
    def wscale(self) -> float:
        """wscale only applies to waveforms; DC-only has none, so always 1.0."""
        return 1.0

    @property
    def pulses(self) -> list:
        return []

    @property
    def pwls(self) -> list:
        return []

    @property
    def static_value(self) -> Optional[float]:
        return None

    @property
    def name(self) -> str:
        """Return name from info.full_name if available."""
        return self.info.full_name if self.info else ''

    def has_waveform_data(self) -> bool:
        """DC-only sources have no waveforms."""
        return False

    def has_current_data(self) -> bool:
        """Check if instance has any current data."""
        return self.dc_value != 0.0

    def get_static_current(self) -> float:
        """Get static/DC current value (mA)."""
        return self.dc_value

    def get_current_at_time(self, t: float) -> float:
        """DC-only: return dc_value (constant regardless of time)."""
        return self.dc_value

    def to_dict(self) -> dict:
        """Serialize for IPC (parallel_parser) and finalize(store_instance_sources=True)."""
        # Extract name from info.full_name if available (for parallel parser compatibility)
        name = self.info.full_name if self.info else ''
        return {
            'name': name,
            'node1': self.node1,
            'node2': self.node2,
            'dc_value': self.dc_value,
            'wscale': 1.0,
            'pulses': [],
            'pwls': [],
            'static_value': None,
            'info': self.info.to_dict() if self.info else None,
        }

    def __repr__(self) -> str:
        return (f"_DCOnlyCurrentSource(node1={self.node1!r}, node2={self.node2!r}, "
                f"dc_value={self.dc_value})")

    def __reduce__(self):
        """Custom pickle support - ensures class is resolved from parser.current_sources module."""
        return (_DCOnlyCurrentSource, (self.node1, self.node2, self.dc_value, self.info))


# =============================================================================
# Parsing Functions
# =============================================================================

def _parse_pulse(pulse_str: str) -> Pulse:
    """Parse pulse definition: pulse(v1, v2, delay, rt, ft, width, period)"""
    match = _RE_PULSE.search(pulse_str)
    if not match:
        return Pulse()

    # Use replace + split instead of re.split for better performance
    content = match.group(1).strip().replace(',', ' ')
    values = content.split()  # split() handles multiple whitespace natively

    pulse = Pulse()
    if len(values) >= 1:
        pulse.v1 = _parse_spice_value(values[0])
    if len(values) >= 2:
        pulse.v2 = _parse_spice_value(values[1])
    if len(values) >= 3:
        pulse.delay = _parse_spice_value(values[2])
    if len(values) >= 4:
        pulse.rt = _parse_spice_value(values[3])
    if len(values) >= 5:
        pulse.ft = _parse_spice_value(values[4])
    if len(values) >= 6:
        pulse.width = _parse_spice_value(values[5])
    if len(values) >= 7:
        pulse.period = _parse_spice_value(values[6])

    return pulse


def _parse_pwl(pwl_str: str) -> PWL:
    """Parse PWL definition: pwl(t1 v1 t2 v2 ...)"""
    match = _RE_PWL.search(pwl_str)
    if not match:
        return PWL()

    # Use replace + split instead of re.split for better performance
    content = match.group(1).strip().replace(',', ' ')
    values = content.split()  # split() handles multiple whitespace natively

    pwl = PWL()
    for i in range(0, len(values) - 1, 2):
        t = _parse_spice_value(values[i])
        v = _parse_spice_value(values[i + 1])
        pwl.points.append((t, v))

    # Only sort if not already sorted (most PWLs are already in time order)
    n = len(pwl.points)
    if n > 1:
        needs_sort = any(pwl.points[i][0] > pwl.points[i+1][0] for i in range(n-1))
        if needs_sort:
            pwl.points.sort(key=lambda x: x[0])
    return pwl


def _parse_current_source_line(line: str) -> Optional[CurrentSource]:
    """
    Parse a current source line and return a CurrentSource object.

    Handles complex formats including:
    - DC values with optional 'dc' prefix
    - static_value= parameter
    - pulse(...) waveforms
    - pwl(...) waveforms with pwl_period= and pwl_delay=
    - sp= (source period) parameter

    When get_optimize_dc_only() is True and the line has no waveforms or
    static_value, returns a lightweight _DCOnlyCurrentSource instead.

    Note: Values are returned in base SPICE units (Amperes). Caller must convert to mA.
    """
    line = line.strip()
    if not line or not line[0].lower() == 'i':
        return None

    # ── Fast path: DC-only with optimize flag ──
    # Check if line contains waveform keywords before expensive tokenization
    if get_optimize_dc_only():
        line_lower = line.lower()
        # Quick check: does line contain waveform keywords or static_value?
        # Check for waveform keywords (without '(' to handle 'pwl (' with space)
        has_waveform = _RE_PWL.search(line_lower) is not None or _RE_PULSE.search(line_lower) is not None

        if not has_waveform and 'static_value=' not in line_lower:
            # Simple split is safe - no parenthesized expressions to preserve
            parts = line.split()
            if len(parts) >= 4:
                name, n1, n2, val_tok = parts[0], parts[1], parts[2], parts[3]
                # Skip 'dc' prefix if present
                if val_tok.lower() == 'dc' and len(parts) >= 5:
                    val_tok = parts[4]
                try:
                    dc_val = _parse_spice_value(val_tok)
                    info = InstanceInfo.parse(name)
                    # Return lightweight DC-only source (wscale not stored - only applies to waveforms)
                    return _DCOnlyCurrentSource(n1, n2, dc_val, info)
                except (ValueError, IndexError):
                    pass  # Fall through to full parser

    # ── Full parser path for waveform sources ──
    # Tokenize preserving parenthesized expressions
    # Use list accumulation instead of string concatenation for O(n) vs O(n²)
    tokens = []
    current = []
    paren_depth = 0

    for char in line:
        if char == '(':
            paren_depth += 1
            current.append(char)
        elif char == ')':
            paren_depth -= 1
            current.append(char)
        elif char in ' \t' and paren_depth == 0:
            if current:
                tokens.append(''.join(current))
                current = []
        else:
            current.append(char)

    if current:
        tokens.append(''.join(current))

    if len(tokens) < 4:
        return None

    isrc = CurrentSource(
        name=tokens[0],
        node1=tokens[1],
        node2=tokens[2]
    )

    # Parse instance name to extract net information
    isrc.info = InstanceInfo.parse(tokens[0])

    # Parse DC value (might be prefixed with 'dc')
    idx = 3
    if tokens[idx].lower() == 'dc' or tokens[idx].lower().startswith('dc.'):
        idx += 1
        if idx < len(tokens):
            try:
                isrc.dc_value = _parse_spice_value(tokens[idx])
                idx += 1
            except ValueError:
                pass
    else:
        try:
            isrc.dc_value = _parse_spice_value(tokens[idx])
            idx += 1
        except ValueError:
            pass

    # Parse remaining parameters
    # Use partition() instead of split('=')[1] to avoid list allocation
    i = idx
    while i < len(tokens):
        token = tokens[i]
        token_lower = token.lower()

        if token_lower.startswith('pulse'):
            # Handle space between 'pulse' and '(' - e.g., 'pulse (...)' becomes two tokens
            pulse_token = token
            if '(' not in token and i + 1 < len(tokens) and tokens[i + 1].startswith('('):
                pulse_token = token + tokens[i + 1]
                i += 1  # Skip the next token since we consumed it
            isrc.pulses.append(_parse_pulse(pulse_token))
        elif token_lower.startswith('pwl') and not token_lower.startswith('pwl_'):
            # Handle space between 'pwl' and '(' - e.g., 'pwl (t1 v1 ...)' becomes two tokens
            pwl_token = token
            if '(' not in token and i + 1 < len(tokens) and tokens[i + 1].startswith('('):
                pwl_token = token + tokens[i + 1]
                i += 1  # Skip the next token since we consumed it
            pwl = _parse_pwl(pwl_token)
            if pwl.points:  # Only add non-empty PWLs
                isrc.pwls.append(pwl)
        elif token_lower.startswith('pwl_period='):
            _, _, value = token.partition('=')
            period = _parse_spice_value(value)
            for pwl in isrc.pwls:
                if pwl.period == 0:
                    pwl.period = period
        elif token_lower.startswith('pwl_delay='):
            _, _, value = token.partition('=')
            delay = _parse_spice_value(value)
            for pwl in isrc.pwls:
                if pwl.delay == 0:
                    pwl.delay = delay
        elif token_lower.startswith('static_value='):
            _, _, value = token.partition('=')
            isrc.static_value = _parse_spice_value(value)
        elif token_lower.startswith('sp='):
            # Source period - apply to pulses
            _, _, value = token.partition('=')
            period = _parse_spice_value(value)
            for pulse in isrc.pulses:
                if pulse.period == 0:
                    pulse.period = period
        elif token_lower.startswith('wscale='):
            # Waveform scaling factor (matches C++ SimPWL pwl_scaling)
            _, _, value = token.partition('=')
            isrc.wscale = _parse_spice_value(value)

        i += 1

    return isrc


# =============================================================================
# PreparedSource & _prepare_instance_source
# =============================================================================

class PreparedSource(NamedTuple):
    """Result of parsing + converting one instance model line."""

    cs: CurrentSource          # Full object with values in mA
    node_pos: str              # Boundary-stripped positive node
    node_neg: str              # Boundary-stripped negative node
    is_boundary_pos: bool      # Had '*' prefix
    is_boundary_neg: bool      # Had '*' prefix
    static_current_ma: float   # cs.get_static_current() (cached)


def _prepare_instance_source(line: str) -> Optional[PreparedSource]:
    """Parse an instance model current source line into mA-converted result.

    Shared pipeline for flat, parallel, and distributed parsers:
    1. _parse_current_source_line(line)
    2. Strip '*' boundary markers from node names
    3. Convert all values from Amperes to mA in-place
    4. Skip zero-current placeholders
    5. Compute static current

    Returns None for invalid lines or lines with no current data.
    """
    cs = _parse_current_source_line(line)
    if cs is None:
        return None

    node_pos = cs.node1
    node_neg = cs.node2
    bnd_pos = node_pos.startswith('*')
    bnd_neg = node_neg.startswith('*')
    if bnd_pos:
        node_pos = node_pos[1:]
    if bnd_neg:
        node_neg = node_neg[1:]
    cs.node1 = node_pos
    cs.node2 = node_neg

    # Convert A → mA
    cs.dc_value *= I_TO_MA
    if cs.static_value is not None:
        cs.static_value *= I_TO_MA
    for pulse in cs.pulses:
        pulse.v1 *= I_TO_MA
        pulse.v2 *= I_TO_MA
    for pwl in cs.pwls:
        pwl.points = [(t, v * I_TO_MA) for t, v in pwl.points]

    if not cs.has_current_data():
        return None

    return PreparedSource(
        cs=cs,
        node_pos=node_pos,
        node_neg=node_neg,
        is_boundary_pos=bnd_pos,
        is_boundary_neg=bnd_neg,
        static_current_ma=cs.get_static_current(),
    )
