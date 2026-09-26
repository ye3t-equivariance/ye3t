"""Config-driven YE3T multiplicity table formatter.

This module promotes the v6 multiplicity-table formatter into the ye3t package.
Use write_latex_table_from_config or write_latex_tables from examples instead of a CLI.
"""

"""
Build YE3T Young--rotation LaTeX tables from multiplicities that have already
been computed or printed elsewhere.

This is deliberately a formatter/normalizer, not a symmetric-group or SO(3)
decomposition engine.  It is therefore safe to use at ranks far beyond the
range where explicit partition enumeration or one-column-per-L tables are
practical.

Accepted inputs
---------------
* CSV, including the legacy rank-6 wide and long CSV exports.
* JSON arrays/objects and JSON Lines.
* Simple printed key=value lines separated by semicolons or tabs.

Compact channel expressions such as

    eta=[1]*16382+[2]*2; l=[1]*16384

are parsed without expanding them into 16,384-element Python lists.  The
complete-channel content is represented internally as run-length encoded
blocks and formatted as

    ((1,1)^16382, (2,1)^2).

Layouts
-------
* auto     : wide for exact ranks N<=8 by default, summary for aggregate rows,
             otherwise exact constant-value L ranges.
* wide     : the historical one-column-per-L layout, now using L rather than L_R.
* ranges   : one row per constant M_{Lambda}^L (or legacy contribution) interval.
* compact  : one row per (kappa, Lambda, lambda) with a compact L profile.
* summary  : one row per reported aggregate result, suited to ranks such as 16384.

The output notation uses L throughout, d_{kappa Lambda} for the product of
block-wise d_b^{kappa_b Lambda_b}, and M_{Lambda}^L for pure angular recoupling
multiplicities.  Legacy input names such as L_R, m^L, and N_Lambda_LR are
accepted only as aliases during import.  Large Young shapes are emitted in
algebraic partition notation; ranks N<=8 use ydiagrams automatically unless
another style is requested.
"""
import ast
import csv
import json
import math
import re
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
__version__ = '6.0.0'


_MISSING = object()


class _FieldSpec:
    def __init__(self, default=_MISSING, default_factory=_MISSING):
        self.default = default
        self.default_factory = default_factory


def field(default=_MISSING, default_factory=_MISSING):
    return _FieldSpec(default=default, default_factory=default_factory)


def _field_default(cls, name):
    if not hasattr(cls, name):
        return _MISSING
    value = getattr(cls, name)
    if isinstance(value, _FieldSpec):
        if value.default_factory is not _MISSING:
            return value.default_factory()
        return value.default
    return value


def dataclass(_cls=None, frozen=False):
    def decorate(cls):
        names = tuple(getattr(cls, '__field_names__', ()))
        if not names:
            names = tuple(
                key for key, value in cls.__dict__.items()
                if not key.startswith('_')
                and not callable(value)
                and not isinstance(value, (staticmethod, classmethod, property))
            )

        def __init__(self, *args, **kwargs):
            if len(args) > len(names):
                raise TypeError(f'{cls.__name__} expected at most {len(names)} positional arguments')
            values = {}
            for name, value in zip(names, args):
                values[name] = value
            for name in names[len(args):]:
                if name in kwargs:
                    values[name] = kwargs.pop(name)
                else:
                    default = _field_default(cls, name)
                    if default is _MISSING:
                        raise TypeError(f'{cls.__name__} missing required argument: {name}')
                    values[name] = default
            if kwargs:
                unknown = next(iter(kwargs))
                raise TypeError(f'{cls.__name__} got unexpected argument: {unknown}')
            for name in names:
                object.__setattr__(self, name, values[name])
            post_init = getattr(self, '__post_init__', None)
            if post_init is not None:
                post_init()

        def __iter_values(self):
            return tuple(getattr(self, name) for name in names)

        def __eq__(self, other):
            return type(self) is type(other) and __iter_values(self) == __iter_values(other)

        def __hash__(self):
            return hash((type(self), __iter_values(self)))

        def __repr__(self):
            body = ', '.join(f'{name}={getattr(self, name)!r}' for name in names)
            return f'{cls.__name__}({body})'

        cls.__field_names__ = names
        cls.__init__ = __init__
        cls.__eq__ = __eq__
        cls.__hash__ = __hash__
        cls.__repr__ = __repr__
        return cls

    if _cls is not None and isinstance(_cls, type):
        return decorate(_cls)
    return decorate


def replace(obj, **changes):
    values = {name: getattr(obj, name) for name in obj.__field_names__}
    values.update(changes)
    return type(obj)(**values)

class TableInputError(ValueError):
    """Raised when external multiplicity data cannot be normalized safely."""

def _is_blank(value):
    return value is None or (isinstance(value, str) and value.strip() == '')

def _first_present(mapping, names):
    for name in names:
        if name in mapping and (not _is_blank(mapping[name])):
            return mapping[name]
    return None

def _as_int(value, *, field_name):
    if isinstance(value, bool):
        raise TableInputError(f'{field_name} must be an integer, not bool')
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace('_', '')
        if re.fullmatch('[+-]?\\d+', text):
            return int(text)
    raise TableInputError(f'Could not parse {field_name}={value!r} as an integer')

def _parse_optional_int(value, *, field_name):
    if _is_blank(value):
        return None
    return _as_int(value, field_name=field_name)

def _make_hashable(value):
    if isinstance(value, list):
        return tuple((_make_hashable(x) for x in value))
    if isinstance(value, tuple):
        return tuple((_make_hashable(x) for x in value))
    if isinstance(value, dict):
        return tuple(sorted(((str(k), _make_hashable(v)) for k, v in value.items())))
    return value

def _split_top_level(text, separators=';'):
    """Split at separators that are not nested in (), [], {}, or quotes."""
    pieces = []
    start = 0
    depth_round = depth_square = depth_curly = 0
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in '\'"':
            quote = char
        elif char == '(':
            depth_round += 1
        elif char == ')':
            depth_round = max(0, depth_round - 1)
        elif char == '[':
            depth_square += 1
        elif char == ']':
            depth_square = max(0, depth_square - 1)
        elif char == '{':
            depth_curly += 1
        elif char == '}':
            depth_curly = max(0, depth_curly - 1)
        elif char in separators and depth_round == depth_square == depth_curly == 0:
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]

def _strip_balanced_outer(text, left, right):
    text = text.strip()
    if not (text.startswith(left) and text.endswith(right)):
        return text
    depth = 0
    quote = None
    escaped = False
    for i, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in '\'"':
            quote = char
        elif char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth == 0 and i != len(text) - 1:
                return text
    return text[1:-1].strip() if depth == 0 else text

def _latex_escape_text(value):
    text = str(value)
    replacements = {'\\': '\\textbackslash{}', '&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#', '_': '\\_', '{': '\\{', '}': '\\}', '~': '\\textasciitilde{}', '^': '\\textasciicircum{}'}
    return ''.join((replacements.get(ch, ch) for ch in text))

def _latex_atom(value):
    if isinstance(value, bool):
        return '\\mathrm{true}' if value else '\\mathrm{false}'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, tuple):
        return '\\left(' + ','.join((_latex_atom(v) for v in value)) + '\\right)'
    return '\\mathrm{' + _latex_escape_text(value) + '}'

def _sanitize_label(label):
    cleaned = re.sub('[^A-Za-z0-9:._-]+', '_', label.strip())
    return cleaned or 'tab:ye3t_external'

def _normalize_tags(value):
    if _is_blank(value):
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(';') if part.strip()]
        return tuple(dict.fromkeys(parts))
    if isinstance(value, Sequence):
        return tuple(dict.fromkeys((str(v).strip() for v in value if str(v).strip())))
    return (str(value),)

@dataclass(frozen=True)
class RLESequence:
    __field_names__ = ('runs', 'original')
    pass
    original = None

    def __post_init__(self):
        clean = []
        for value, count in self.runs:
            count = _as_int(count, field_name='run count')
            if count < 0:
                raise TableInputError('Sequence run counts must be nonnegative')
            if count == 0:
                continue
            value = _make_hashable(value)
            if clean and clean[-1][0] == value:
                clean[-1] = (value, clean[-1][1] + count)
            else:
                clean.append((value, count))
        object.__setattr__(self, 'runs', tuple(clean))

    @property
    def length(self):
        return sum((count for _, count in self.runs))

    def concat(self, other):
        return RLESequence(self.runs + other.runs)

    def repeat(self, times, *, max_runs=200000):
        times = _as_int(times, field_name='sequence repetition')
        if times < 0:
            raise TableInputError('Sequence repetition must be nonnegative')
        if times == 0 or not self.runs:
            return RLESequence(())
        if times == 1:
            return self
        if len(self.runs) == 1:
            value, count = self.runs[0]
            return RLESequence(((value, count * times),))
        boundary_merges = 1 if self.runs[-1][0] == self.runs[0][0] else 0
        estimated = len(self.runs) * times - boundary_merges * (times - 1)
        if estimated > max_runs:
            raise TableInputError("The repeated eta/l pattern would create too many run blocks. Supply compact complete-channel blocks through the 'nu' field instead.")
        return RLESequence(self.runs * times)

    def compact_text(self):
        if not self.runs:
            return '[]'
        pieces = []
        for value, count in self.runs:
            atom = repr(value)
            if count == 1:
                pieces.append(f'[{atom}]')
            else:
                pieces.append(f'[{atom}]*{count}')
        return '+'.join(pieces)

    def to_jsonable(self):
        return {'runs': [[value, count] for value, count in self.runs], 'length': self.length, 'compact': self.compact_text()}

def _sequence_from_ast(node):
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for element in node.elts:
            try:
                values.append(_make_hashable(ast.literal_eval(element)))
            except Exception as exc:
                raise TableInputError('eta/l list elements must be literal numbers, strings, or tuples') from exc
        return RLESequence(tuple(((value, 1) for value in values)))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sequence_from_ast(node.left).concat(_sequence_from_ast(node.right))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, int):
            return _sequence_from_ast(node.right).repeat(node.left.value)
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
            return _sequence_from_ast(node.left).repeat(node.right.value)
        raise TableInputError('Only integer repetition is allowed in compact eta/l expressions')
    if isinstance(node, ast.Constant):
        return RLESequence(((_make_hashable(node.value), 1),))
    raise TableInputError('Unsupported eta/l expression. Use literals plus list concatenation/repetition, for example [1]*16382+[2]*2.')

def parse_sequence_spec(value, *, field_name):
    if _is_blank(value):
        return None
    if isinstance(value, RLESequence):
        return value
    if isinstance(value, Mapping):
        runs = value.get('runs')
        if runs is None:
            raise TableInputError(f"{field_name} mapping must contain a 'runs' key")
        return RLESequence(tuple(((_make_hashable(item[0]), item[1]) for item in runs)))
    if isinstance(value, (list, tuple)):
        return RLESequence(tuple(((_make_hashable(item), 1) for item in value)))
    if isinstance(value, (int, float)):
        return RLESequence(((_make_hashable(value), 1),))
    text = str(value).strip()
    try:
        node = ast.parse(text, mode='eval').body
    except SyntaxError as exc:
        raise TableInputError(f'Could not parse {field_name} expression {text!r}') from exc
    seq = _sequence_from_ast(node)
    return RLESequence(seq.runs, original=text)

@dataclass(frozen=True)
class ChannelBlock:
    __field_names__ = ('eta', 'ell', 'count')
    pass
    pass
    pass

    def key(self):
        return (_make_hashable(self.eta), _make_hashable(self.ell), self.count)

def derive_channel_blocks(eta, ell):
    if eta.length != ell.length:
        raise TableInputError(f'eta and l have different lengths ({eta.length} != {ell.length})')
    counts = OrderedDict()
    i = j = 0
    rem_eta = eta.runs[0][1] if eta.runs else 0
    rem_ell = ell.runs[0][1] if ell.runs else 0
    while i < len(eta.runs) and j < len(ell.runs):
        value_eta = eta.runs[i][0]
        value_ell = ell.runs[j][0]
        take = min(rem_eta, rem_ell)
        pair = (_make_hashable(value_eta), _make_hashable(value_ell))
        counts[pair] = counts.get(pair, 0) + take
        rem_eta -= take
        rem_ell -= take
        if rem_eta == 0:
            i += 1
            if i < len(eta.runs):
                rem_eta = eta.runs[i][1]
        if rem_ell == 0:
            j += 1
            if j < len(ell.runs):
                rem_ell = ell.runs[j][1]
    return tuple((ChannelBlock(a, l, count) for (a, l), count in counts.items()))

def parse_nu_spec(value):
    if _is_blank(value):
        return None
    if isinstance(value, Mapping):
        value = value.get('blocks', value.get('nu', value))
    if isinstance(value, Sequence) and (not isinstance(value, str)):
        blocks = []
        for item in value:
            if isinstance(item, Mapping):
                eta = _first_present(item, ('eta', 'n', 'channel'))
                ell = _first_present(item, ('l', 'ell'))
                count = _first_present(item, ('count', 'k', 'multiplicity'))
            elif isinstance(item, Sequence) and len(item) == 3:
                eta, ell, count = item
            else:
                raise TableInputError('Each nu block must be a mapping or a triple [eta, l, count]')
            blocks.append(ChannelBlock(_make_hashable(eta), _make_hashable(ell), _as_int(count, field_name='nu count')))
        return tuple(blocks)
    text = str(value).strip()
    pieces = _split_top_level(text, separators=';')
    blocks = []
    for piece in pieces:
        match = re.fullmatch('\\s*\\((.+),(.+)\\)\\s*(?:\\^|\\*|x|×)?\\s*(\\d+)?\\s*', piece)
        if not match:
            raise TableInputError("Could not parse nu block. Use JSON triples or '(eta,l)^count; ...'.")
        eta_text, ell_text, count_text = match.groups()
        try:
            eta = _make_hashable(ast.literal_eval(eta_text.strip()))
        except Exception:
            eta = eta_text.strip()
        try:
            ell = _make_hashable(ast.literal_eval(ell_text.strip()))
        except Exception:
            ell = ell_text.strip()
        blocks.append(ChannelBlock(eta, ell, int(count_text or 1)))
    return tuple(blocks)

@dataclass(frozen=True)
class PartitionSpec:
    __field_names__ = ('runs', 'special')
    runs = ()
    special = None

    def __post_init__(self):
        if self.special is not None:
            object.__setattr__(self, 'runs', ())
            return
        clean = []
        previous = None
        for part, count in self.runs:
            part = _as_int(part, field_name='partition part')
            count = _as_int(count, field_name='partition exponent')
            if part <= 0 or count <= 0:
                raise TableInputError('Partition parts and exponents must be positive')
            if previous is not None and part > previous:
                raise TableInputError('Partition parts must be in nonincreasing order')
            if clean and clean[-1][0] == part:
                clean[-1] = (part, clean[-1][1] + count)
            else:
                clean.append((part, count))
            previous = part
        object.__setattr__(self, 'runs', tuple(clean))

    @property
    def total(self):
        if self.special is not None:
            return None
        return sum((part * count for part, count in self.runs))

    @property
    def number_of_rows(self):
        if self.special is not None:
            return None
        return sum((count for _, count in self.runs))

    def expanded(self, *, max_items=256):
        if self.special is not None:
            raise TableInputError('A special partition label cannot be expanded')
        nrows = self.number_of_rows or 0
        if nrows > max_items:
            raise TableInputError('Partition is too large to expand safely')
        return tuple((part for part, count in self.runs for _ in range(count)))

    def is_one_row(self):
        return self.special is None and len(self.runs) == 1 and (self.runs[0][1] == 1)

    def is_one_column(self):
        return self.special is None and len(self.runs) == 1 and (self.runs[0][0] == 1)

    def compact_text(self):
        if self.special is not None:
            return self.special
        terms = [str(part) if count == 1 else f'{part}^{count}' for part, count in self.runs]
        return '(' + ','.join(terms) + ')'

    def to_jsonable(self):
        if self.special is not None:
            return self.special
        return {'runs': [[part, count] for part, count in self.runs], 'total': self.total}

def parse_partition(value, *, field_name='partition'):
    if _is_blank(value):
        return None
    if isinstance(value, PartitionSpec):
        return value
    if isinstance(value, Mapping):
        if 'latex' in value or 'raw_latex' in value:
            raw = value.get('latex', value.get('raw_latex'))
            return PartitionSpec(special='latex:' + str(raw))
        if 'special' in value:
            return PartitionSpec(special=str(value['special']))
        if 'runs' in value:
            return PartitionSpec(tuple(((_as_int(p, field_name=field_name), _as_int(c, field_name=field_name)) for p, c in value['runs'])))
        if 'parts' in value:
            value = value['parts']
        else:
            raise TableInputError(f'{field_name} mapping must contain runs, parts, or special')
    if isinstance(value, int):
        return PartitionSpec(((value, 1),))
    if isinstance(value, Sequence) and (not isinstance(value, str)):
        parts = [_as_int(v, field_name=field_name) for v in value]
        runs = []
        for part in parts:
            if runs and runs[-1][0] == part:
                runs[-1] = (part, runs[-1][1] + 1)
            else:
                runs.append((part, 1))
        return PartitionSpec(tuple(runs))
    text = str(value).strip()
    lower = text.lower()
    if lower.startswith('latex:') or lower.startswith('tex:'):
        return PartitionSpec(special='latex:' + text.split(':', 1)[1].strip())
    if lower in {'all', 'any', '*', 'summed'}:
        return PartitionSpec(special='all')
    if lower in {'trivial', 'symmetric', 'ace', 'bosonic'}:
        return PartitionSpec(special='trivial')
    if lower in {'sign', 'antisymmetric', 'fermionic', 'sgn'}:
        return PartitionSpec(special='sign')
    if lower in {'--', 'none', 'not resolved', 'unresolved'}:
        return PartitionSpec(special='unresolved')
    text = re.sub('\\\\left|\\\\right', '', text)
    ydiag = re.fullmatch('\\\\ydiagram\\{(.+)\\}', text)
    if ydiag:
        text = ydiag.group(1)
    text = _strip_balanced_outer(text, '(', ')')
    text = _strip_balanced_outer(text, '[', ']')
    if '*' in text and text.strip().startswith('['):
        seq = parse_sequence_spec(text, field_name=field_name)
        assert seq is not None
        runs = tuple(((_as_int(part, field_name=field_name), count) for part, count in seq.runs))
        return PartitionSpec(runs)
    if not text:
        return PartitionSpec(())
    terms = _split_top_level(text, separators=',')
    runs = []
    for term in terms:
        term = term.strip().replace('{', '').replace('}', '')
        match = re.fullmatch('(\\d+)\\s*(?:\\^\\s*(\\d+))?', term)
        if not match:
            raise TableInputError(f'Could not parse {field_name} term {term!r}; use e.g. (4,2), (1^6), or [1]*6')
        part, exponent = match.groups()
        runs.append((int(part), int(exponent or 1)))
    return PartitionSpec(tuple(runs))

@dataclass(frozen=True)
class KappaSpec:
    __field_names__ = ('partitions', 'special')
    partitions = ()
    special = None

    def to_jsonable(self):
        if self.special is not None:
            return self.special
        return [partition.to_jsonable() for partition in self.partitions]

def parse_kappa(value):
    if _is_blank(value):
        return None
    if isinstance(value, KappaSpec):
        return value
    if isinstance(value, Mapping) and 'special' in value:
        return KappaSpec(special=str(value['special']))
    if isinstance(value, Sequence) and (not isinstance(value, str)):
        if all((isinstance(item, (int, float)) for item in value)):
            partition = parse_partition(value, field_name='kappa')
            assert partition is not None
            return KappaSpec((partition,))
        parts = tuple((parse_partition(item, field_name='kappa block') for item in value))
        return KappaSpec(tuple((part for part in parts if part is not None)))
    text = str(value).strip()
    lower = text.lower()
    if lower in {'all', 'any', '*', 'summed'}:
        return KappaSpec(special='all')
    if lower in {'trivial', 'symmetric', 'ace', 'bosonic'}:
        return KappaSpec(special='trivial')
    if lower in {'sign', 'antisymmetric', 'fermionic', 'sgn'}:
        return KappaSpec(special='sign')
    if lower in {'--', 'none', 'not resolved', 'unresolved'}:
        return KappaSpec(special='unresolved')
    text = re.sub('\\\\left|\\\\right', '', text)
    text = _strip_balanced_outer(text, '(', ')')
    pieces = _split_top_level(text, separators='|')
    if not pieces:
        raise TableInputError('Empty kappa specification')
    partitions = tuple((parse_partition(piece, field_name='kappa block') for piece in pieces))
    return KappaSpec(tuple((part for part in partitions if part is not None)))

@dataclass(frozen=True)
class LambdaSpec:
    __field_names__ = ('values', 'special')
    values = ()
    special = None

    def to_jsonable(self):
        return self.special if self.special is not None else list(self.values)

def parse_block_lambda(value):
    if _is_blank(value):
        return None
    if isinstance(value, LambdaSpec):
        return value
    if isinstance(value, Mapping) and 'special' in value:
        return LambdaSpec(special=str(value['special']))
    if isinstance(value, Sequence) and (not isinstance(value, str)):
        return LambdaSpec(tuple((_as_int(v, field_name='Lambda') for v in value)))
    if isinstance(value, (int, float)):
        return LambdaSpec((_as_int(value, field_name='Lambda'),))
    text = str(value).strip()
    lower = text.lower()
    if lower in {'all', 'any', '*', 'summed', 'sum'}:
        return LambdaSpec(special='all')
    if lower in {'--', 'none', 'not resolved', 'unresolved'}:
        return LambdaSpec(special='unresolved')
    text = re.sub('\\\\left|\\\\right', '', text)
    text = _strip_balanced_outer(text, '(', ')')
    text = _strip_balanced_outer(text, '[', ']')
    if not text:
        return LambdaSpec(())
    return LambdaSpec(tuple((_as_int(piece, field_name='Lambda') for piece in _split_top_level(text, separators=','))))

@dataclass(frozen=True)
class LSegment:
    __field_names__ = ('start', 'stop', 'step', 'multiplicity')
    pass
    pass
    step = 1
    multiplicity = None

    def __post_init__(self):
        start = _as_int(self.start, field_name='L start')
        stop = _as_int(self.stop, field_name='L stop')
        step = _as_int(self.step, field_name='L step')
        if start < 0 or stop < 0:
            raise TableInputError('Angular momenta L must be nonnegative')
        if stop < start:
            raise TableInputError('L range stop must not be below its start')
        if step <= 0:
            raise TableInputError('L range step must be positive')
        object.__setattr__(self, 'start', start)
        object.__setattr__(self, 'stop', stop)
        object.__setattr__(self, 'step', step)
        if self.multiplicity is not None:
            object.__setattr__(self, 'multiplicity', _as_int(self.multiplicity, field_name='multiplicity'))

    @property
    def count(self):
        return (self.stop - self.start) // self.step + 1

    def iter_values(self, *, limit=None):
        if limit is not None and self.count > limit:
            raise TableInputError(f'Refusing to expand an L range with {self.count} values (limit={limit})')
        yield from range(self.start, self.stop + 1, self.step)

    def to_jsonable(self):
        out = {'min': self.start, 'max': self.stop, 'step': self.step}
        if self.multiplicity is not None:
            out['multiplicity'] = self.multiplicity
        return out

@dataclass
class MultiplicityProfile:
    __field_names__ = ('segments', 'total_override', 'reported_value')
    segments = field(default_factory=list)
    total_override = None
    reported_value = None

    def known(self):
        return bool(self.segments) and all((segment.multiplicity is not None for segment in self.segments))

    def support_segments(self):
        return [replace(segment, multiplicity=None) for segment in self.segments]

    def computed_total(self):
        if not self.known():
            return None
        return sum((segment.count * int(segment.multiplicity or 0) for segment in self.segments))

    def total(self):
        if self.total_override is not None:
            return self.total_override
        return self.computed_total()

    def max_L(self):
        return max((segment.stop for segment in self.segments), default=None)

    def min_L(self):
        return min((segment.start for segment in self.segments), default=None)

    def to_points(self, *, limit=100000):
        if not self.known():
            raise TableInputError('Cannot expand an L support range without per-L multiplicities')
        total_values = sum((segment.count for segment in self.segments))
        if total_values > limit:
            raise TableInputError(f'Profile contains {total_values} L values, exceeding expansion limit {limit}')
        points = defaultdict(int)
        for segment in self.segments:
            assert segment.multiplicity is not None
            for L in segment.iter_values():
                points[L] += segment.multiplicity
        return dict(sorted(points.items()))

    def to_jsonable(self):
        out = {'L_ranges': [segment.to_jsonable() for segment in self.segments]}
        if self.total_override is not None:
            out['total_mult'] = self.total_override
        if self.reported_value is not None:
            out['reported_value'] = self.reported_value
        return out

def compress_L_points(points):
    items = sorted(((int(L), int(mult)) for L, mult in points.items() if int(mult) != 0))
    if not items:
        return []
    segments = []
    run_L = [items[0][0]]
    run_mult = items[0][1]
    run_step = None

    def flush():
        nonlocal run_L, run_mult, run_step
        if not run_L:
            return
        segments.append(LSegment(run_L[0], run_L[-1], run_step or 1, run_mult))
    for L, mult in items[1:]:
        gap = L - run_L[-1]
        compatible = mult == run_mult and gap > 0 and (run_step is None or gap == run_step)
        if compatible:
            if run_step is None:
                run_step = gap
            run_L.append(L)
        else:
            flush()
            run_L = [L]
            run_mult = mult
            run_step = None
    flush()
    return segments

def compress_L_support(values):
    unique = sorted(set((_as_int(v, field_name='L') for v in values)))
    if not unique:
        return []
    segments = []
    run = [unique[0]]
    step = None

    def flush():
        nonlocal run, step
        segments.append(LSegment(run[0], run[-1], step or 1, None))
    for L in unique[1:]:
        gap = L - run[-1]
        compatible = gap > 0 and (step is None or gap == step)
        if compatible:
            if step is None:
                step = gap
            run.append(L)
        else:
            flush()
            run = [L]
            step = None
    flush()
    return segments

def _parse_L_range_text(text):
    text = text.strip().replace('≤', '<=').replace('…', '...')
    match = re.fullmatch('(\\d+)\\s*<=\\s*L\\s*<=\\s*(\\d+)', text, flags=re.I)
    if match:
        return [LSegment(int(match.group(1)), int(match.group(2)))]
    match = re.fullmatch('range\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*(?:,\\s*(\\d+)\\s*)?\\)', text)
    if match:
        start, exclusive, step = (int(match.group(1)), int(match.group(2)), int(match.group(3) or 1))
        stop = exclusive - 1
        while stop >= start and (stop - start) % step:
            stop -= 1
        return [] if stop < start else [LSegment(start, stop, step)]
    match = re.fullmatch('(\\d+)\\s*\\.\\.\\s*(\\d+)(?:\\s*/\\s*(\\d+))?', text)
    if match:
        return [LSegment(int(match.group(1)), int(match.group(2)), int(match.group(3) or 1))]
    match = re.fullmatch('(\\d+)\\s*-\\s*(\\d+)(?:\\s*/\\s*(\\d+))?', text)
    if match:
        return [LSegment(int(match.group(1)), int(match.group(2)), int(match.group(3) or 1))]
    match = re.fullmatch('(\\d+)\\s*:\\s*(\\d+)\\s*:\\s*(\\d+)', text)
    if match:
        start, step, stop = map(int, match.groups())
        return [LSegment(start, stop, step)]
    match = re.fullmatch('(\\d+)\\s*:\\s*(\\d+)', text)
    if match:
        return [LSegment(int(match.group(1)), int(match.group(2)))]
    return None

def parse_L_support(value):
    if _is_blank(value):
        return []
    if isinstance(value, LSegment):
        return [value]
    if isinstance(value, Mapping):
        if any((key in value for key in ('min', 'max', 'start', 'stop'))):
            start = _first_present(value, ('min', 'start', 'L_min'))
            stop = _first_present(value, ('max', 'stop', 'L_max'))
            step = _first_present(value, ('step', 'L_step')) or 1
            mult_value = _first_present(value, ('M', 'M_Lambda_L', 'alpha', 'multiplicity', 'mult', 'm', 'contribution', 'value'))
            multiplicity = None if _is_blank(mult_value) else _as_int(mult_value, field_name='multiplicity')
            return [LSegment(_as_int(start, field_name='L min'), _as_int(stop, field_name='L max'), _as_int(step, field_name='L step'), multiplicity)]
        raise TableInputError('L range mapping must contain min/max or start/stop')
    if isinstance(value, (int, float)):
        L = _as_int(value, field_name='L')
        return [LSegment(L, L)]
    if isinstance(value, Sequence) and (not isinstance(value, str)):
        if value and all((isinstance(item, Mapping) for item in value)):
            out = []
            for item in value:
                out.extend(parse_L_support(item))
            return out
        return compress_L_support((_as_int(item, field_name='L') for item in value))
    text = str(value).strip()
    parsed_range = _parse_L_range_text(text)
    if parsed_range is not None:
        return parsed_range
    try:
        literal = ast.literal_eval(text)
    except Exception:
        literal = None
    if literal is not None and literal != text:
        return parse_L_support(literal)
    if ',' in text:
        return compress_L_support((_as_int(piece, field_name='L') for piece in _split_top_level(text, separators=',')))
    if re.fullmatch('\\d+', text):
        L = int(text)
        return [LSegment(L, L)]
    raise TableInputError(f'Could not parse L support {value!r}')

def parse_L_mults(value):
    if _is_blank(value):
        return {}
    if isinstance(value, Mapping):
        return {_as_int(L, field_name='L'): _as_int(mult, field_name='multiplicity') for L, mult in value.items() if _as_int(mult, field_name='multiplicity') != 0}
    if isinstance(value, Sequence) and (not isinstance(value, str)):
        points = {}
        for item in value:
            if isinstance(item, Mapping):
                L = _first_present(item, ('L', 'L_R', 'ell'))
                mult = _first_present(item, ('M', 'M_Lambda_L', 'alpha', 'multiplicity', 'mult', 'm', 'contribution', 'value'))
                points[_as_int(L, field_name='L')] = _as_int(mult, field_name='multiplicity')
            elif isinstance(item, Sequence) and len(item) == 2:
                points[_as_int(item[0], field_name='L')] = _as_int(item[1], field_name='multiplicity')
            else:
                raise TableInputError('L_mults list entries must be [L,m] pairs or mappings')
        return points
    text = str(value).strip()
    try:
        literal = ast.literal_eval(text)
    except Exception:
        literal = None
    if isinstance(literal, Mapping):
        return parse_L_mults(literal)
    points = {}
    for piece in _split_top_level(text, separators=','):
        match = re.fullmatch('\\s*(\\d+)\\s*[:=]\\s*([+-]?\\d+)\\s*', piece)
        if not match:
            raise TableInputError("Could not parse L_mults. Use a mapping such as {0:1,1:2} or '0:1,1:2'.")
        points[int(match.group(1))] = int(match.group(2))
    return points

@dataclass
class TableRecord:
    __field_names__ = (
        'N', 'eta', 'ell', 'nu_blocks', 'kappa', 'block_lambda',
        'parent_lambda', 'profile', 'c_value', 'd_value', 'd_blocks',
        'profile_kind', 'code_counts', 'count_scope', 'tags', 'row_color',
        'row_id', 'source_index', 'metadata'
    )
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    c_value = None
    d_value = None
    d_blocks = None
    profile_kind = 'M'
    code_counts = field(default_factory=OrderedDict)
    count_scope = None
    tags = ()
    row_color = None
    row_id = None
    source_index = 0
    metadata = field(default_factory=dict)

    def grouping_key(self):
        return (self.N, tuple((block.key() for block in self.nu_blocks)), self.kappa, self.block_lambda, self.parent_lambda, self.c_value, self.d_value, self.d_blocks, self.profile_kind, self.count_scope, self.tags, self.row_color, self.row_id)

    def derived_row_count(self):
        total = self.profile.total()
        if total is None:
            return None
        if self.profile_kind == 'M':
            c_factor = 1 if self.c_value is None else self.c_value
            d_factor = 1 if self.d_value is None else self.d_value
            return total * c_factor * d_factor
        return total

    def to_jsonable(self):
        out = {'N': self.N, 'nu': [{'eta': block.eta, 'l': block.ell, 'count': block.count} for block in self.nu_blocks], 'kappa': None if self.kappa is None else self.kappa.to_jsonable(), 'Lambda': None if self.block_lambda is None else self.block_lambda.to_jsonable(), 'lambda': None if self.parent_lambda is None else self.parent_lambda.to_jsonable(), 'profile_kind': self.profile_kind, 'counts': dict(self.code_counts), 'tags': list(self.tags)}
        if self.count_scope is not None:
            out['count_scope'] = self.count_scope
        if self.eta is not None:
            out['eta'] = self.eta.to_jsonable()
        if self.ell is not None:
            out['l'] = self.ell.to_jsonable()
        if self.c_value is not None:
            out['c_kappa_lambda'] = self.c_value
        if self.d_value is not None:
            out['d_kappa_Lambda'] = self.d_value
        if self.d_blocks is not None:
            out['d_blocks'] = list(self.d_blocks)
        if self.profile.segments:
            if self.profile_kind == 'M':
                range_key, value_key = ('M_ranges', 'M')
            elif self.profile_kind == 'alpha':
                range_key, value_key = ('alpha_ranges', 'alpha')
            else:
                range_key, value_key = ('contribution_ranges', 'contribution')
            ranges = []
            for segment in self.profile.segments:
                item = {'min': segment.start, 'max': segment.stop, 'step': segment.step}
                if segment.multiplicity is not None:
                    item[value_key] = segment.multiplicity
                ranges.append(item)
            if self.profile.known():
                out[range_key] = ranges
            else:
                out['L_support'] = ranges
        if self.profile.total_override is not None:
            out['profile_total'] = self.profile.total_override
        if self.profile.reported_value is not None:
            out['reported_value'] = self.profile.reported_value
        if self.row_color is not None:
            out['row_color'] = self.row_color
        if self.row_id is not None:
            out['row_id'] = self.row_id
        return out
ETA_ALIASES = ('eta', 'eta_in', 'eta_input', 'boldsymbol_eta')
ELL_ALIASES = ('l_in', 'ell_in', 'l_input', 'ells', 'ell', 'l')
NU_ALIASES = ('nu', 'nu_blocks', 'channel_blocks', 'complete_channels')
KAPPA_ALIASES = ('kappa', 'kappa_tuple', 'boldsymbol_kappa')
BLOCK_LAMBDA_ALIASES = ('Lambda', 'Lambda_tuple', 'boldsymbol_Lambda', 'block_Lambda')
PARENT_LAMBDA_ALIASES = ('lambda', 'lam', 'parent_lambda')
C_ALIASES = ('c_kappa_lambda', 'c', 'branching_multiplicity')
D_ALIASES = ('d_kappa_Lambda', 'd_kappaLambda', 'd_block_product', 'd', 'block_multiplicity')
D_BLOCKS_ALIASES = ('d_blocks', 'block_d', 'd_b_values', 'block_multiplicities')
TOTAL_ALIASES = ('total_mult', 'total', 'sum_multiplicity', 'profile_total', 'row_total')
REPORTED_ALIASES = ('reported_value', 'value', 'reported_count')
COUNTS_ALIASES = ('counts', 'code_counts', 'external_counts', 'comparison_counts')
PROFILE_KIND_ALIASES = ('profile_kind', 'L_value_kind', 'profile_value_kind')
COUNT_SCOPE_ALIASES = ('count_scope', 'D_scope', 'total_scope')
ROW_COLOR_ALIASES = ('row_color', 'color', 'latex_row_color')
L_SCALAR_ALIASES = ('L', 'L_R', 'final_L')
L_SUPPORT_ALIASES = ('M_ranges', 'M_Lambda_L_ranges', 'alpha_ranges', 'contribution_ranges', 'L_support', 'allowed_L', 'L_range', 'L_ranges')
M_PROFILE_ALIASES = ('M_Lambda_L', 'M_by_L', 'M_mults', 'M_values', 'recoupling_multiplicities')
ALPHA_PROFILE_ALIASES = ('alpha_by_L', 'alpha_L', 'alpha_mults', 'parent_sector_multiplicities')
GENERIC_PROFILE_ALIASES = ('L_mults', 'multiplicities', 'profile', 'm_by_L')
M_PER_L_ALIASES = ('M_Lambda_L_value', 'M_per_L', 'M', 'recoupling_multiplicity')
ALPHA_PER_L_ALIASES = ('alpha', 'alpha_value', 'alpha_per_L', 'parent_sector_multiplicity')
CONTRIBUTION_PER_L_ALIASES = ('contribution', 'alpha_contribution', 'row_contribution', 'multiplicity', 'm', 'm_L')
KNOWN_COUNT_CODES = ('YE3T', 'e3nn', 'cueq', 'GEPI')

def _normalized_key_lookup(raw):
    return {str(key).strip(): str(key) for key in raw}

def _get_alias(raw, aliases):
    for alias in aliases:
        if alias in raw and (not _is_blank(raw[alias])):
            return raw[alias]
    lower_to_keys = defaultdict(list)
    for key in raw:
        lower_to_keys[str(key).lower()].append(key)
    for alias in aliases:
        if len(alias) <= 1 or alias in {'Lambda', 'lambda'}:
            continue
        keys = lower_to_keys.get(alias.lower(), [])
        if len(keys) == 1 and (not _is_blank(raw[keys[0]])):
            return raw[keys[0]]
    return None

def _get_alias_with_name(raw, aliases):
    for alias in aliases:
        if alias in raw and (not _is_blank(raw[alias])):
            return (alias, raw[alias])
    lower_to_keys = defaultdict(list)
    for key in raw:
        lower_to_keys[str(key).lower()].append(key)
    for alias in aliases:
        if len(alias) <= 1 or alias in {'Lambda', 'lambda'}:
            continue
        keys = lower_to_keys.get(alias.lower(), [])
        if len(keys) == 1 and (not _is_blank(raw[keys[0]])):
            return (str(keys[0]), raw[keys[0]])
    return (None, None)

def _parse_count_value(value):
    if _is_blank(value):
        return None
    try:
        return _as_int(value, field_name='code count')
    except TableInputError:
        return str(value).strip()

def _parse_counts_mapping(value):
    out = OrderedDict()
    if _is_blank(value):
        return out
    if isinstance(value, Mapping):
        for code, count in value.items():
            out[str(code).strip()] = _parse_count_value(count)
        return out
    text = str(value).strip()
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
    if isinstance(parsed, Mapping):
        return _parse_counts_mapping(parsed)
    for piece in _split_top_level(text, separators=','):
        if ':' in piece:
            code, count = piece.split(':', 1)
        elif '=' in piece:
            code, count = piece.split('=', 1)
        else:
            raise TableInputError('Counts must be a mapping or comma-separated code:value pairs')
        out[code.strip()] = _parse_count_value(count.strip())
    return out

def _canonical_count_code(name):
    for code in KNOWN_COUNT_CODES:
        if name.lower() == code.lower():
            return code
    return name

def _extract_code_counts(raw):
    counts = _parse_counts_mapping(_get_alias(raw, COUNTS_ALIASES))
    canonical = OrderedDict()
    for code, value in counts.items():
        canonical[_canonical_count_code(code)] = value
    for key, value in raw.items():
        if _is_blank(value):
            continue
        name = str(key).strip()
        matched_code = None
        for code in KNOWN_COUNT_CODES:
            if name.lower() == code.lower():
                matched_code = code
                break
        if matched_code is None:
            match = re.fullmatch('count[._-](.+)', name, flags=re.I)
            if match:
                matched_code = match.group(1)
            else:
                match = re.fullmatch('(.+?)[._-]count', name, flags=re.I)
                if match:
                    matched_code = match.group(1)
        if matched_code:
            canonical[_canonical_count_code(matched_code)] = _parse_count_value(value)
    reported = _get_alias(raw, REPORTED_ALIASES)
    if reported is not None and 'YE3T' not in canonical:
        canonical['YE3T'] = _parse_count_value(reported)
    return canonical

def _parse_int_tuple(value, *, field_name):
    if _is_blank(value):
        return None
    if isinstance(value, Mapping):
        value = value.get('values', value.get('blocks', value))
    if isinstance(value, Sequence) and (not isinstance(value, str)):
        return tuple((_as_int(item, field_name=field_name) for item in value))
    text = str(value).strip()
    try:
        literal = ast.literal_eval(text)
    except Exception:
        literal = None
    if isinstance(literal, (list, tuple)):
        return tuple((_as_int(item, field_name=field_name) for item in literal))
    text = _strip_balanced_outer(text, '(', ')')
    text = _strip_balanced_outer(text, '[', ']')
    return tuple((_as_int(piece, field_name=field_name) for piece in _split_top_level(text, separators=',')))

def _normalize_profile_kind(value, *, default='M'):
    if _is_blank(value):
        return default
    text = str(value).strip().lower().replace('-', '_')
    if text in {'m', 'recoupling', 'angular', 'm_lambda_l', 'pure_m'}:
        return 'M'
    if text in {'alpha', 'parent', 'parent_sector', 'alpha_lambda_l'}:
        return 'alpha'
    if text in {'contribution', 'weighted', 'legacy', 'c_d_m', 'cdm'}:
        return 'contribution'
    raise TableInputError("profile_kind must be 'M', 'alpha', or 'contribution'")

def _normalize_count_scope(value):
    if _is_blank(value) or str(value).strip().lower() == 'auto':
        return None
    text = str(value).strip().lower().replace('-', '_')
    aliases = {'lambda': 'lambda', 'sector': 'lambda', 'per_lambda': 'lambda', 'global': 'global', 'all_lambda': 'global', 'all': 'global', 'row': 'row', 'detailed': 'row', 'kappa_lambda': 'row'}
    if text not in aliases:
        raise TableInputError('count_scope must be auto, lambda, global, or row')
    return aliases[text]

def _extract_wide_L_points(raw):
    points = {}
    for key, value in raw.items():
        if _is_blank(value):
            continue
        match = re.fullmatch('L_?(\\d+)', str(key).strip())
        if not match:
            continue
        mult = _as_int(value, field_name=str(key))
        if mult:
            points[int(match.group(1))] = mult
    return points

def _parse_reported_value(value):
    if _is_blank(value):
        return None
    try:
        return _as_int(value, field_name='reported value')
    except TableInputError:
        return str(value).strip()

def _infer_N(raw, eta, ell, blocks, parent_lambda):
    explicit = _get_alias(raw, ('N', 'rank', 'order'))
    candidates = []
    if explicit is not None:
        candidates.append(('N', _as_int(explicit, field_name='N')))
    if eta is not None:
        candidates.append(('eta', eta.length))
    if ell is not None:
        candidates.append(('l', ell.length))
    if blocks:
        candidates.append(('nu', sum((block.count for block in blocks))))
    if parent_lambda is not None and parent_lambda.total is not None:
        candidates.append(('lambda', parent_lambda.total))
    if not candidates:
        raise TableInputError('N could not be inferred; supply N, eta/l, nu, or lambda')
    N = candidates[0][1]
    conflicts = [(name, value) for name, value in candidates if value != N]
    if conflicts:
        details = ', '.join((f'{name}={value}' for name, value in candidates))
        raise TableInputError(f'Inconsistent rank information: {details}')
    return N

def _trivial_kappa(blocks):
    return KappaSpec(tuple((PartitionSpec(((block.count, 1),)) for block in blocks)))

def _sign_kappa(blocks):
    return KappaSpec(tuple((PartitionSpec(((1, block.count),)) for block in blocks)))

def _trivial_parent(N):
    return PartitionSpec(((N, 1),))

def _sign_parent(N):
    return PartitionSpec(((1, N),))

def _validate_record(record):
    if record.N <= 0:
        raise TableInputError('N must be positive')
    if record.nu_blocks and sum((block.count for block in record.nu_blocks)) != record.N:
        raise TableInputError('nu block counts do not sum to N')
    if record.kappa is not None and record.kappa.special is None and record.nu_blocks:
        if len(record.kappa.partitions) != len(record.nu_blocks):
            raise TableInputError(f'kappa must contain one partition per complete-channel block ({len(record.kappa.partitions)} != {len(record.nu_blocks)})')
        for index, (partition, block) in enumerate(zip(record.kappa.partitions, record.nu_blocks), start=1):
            if partition.total is not None and partition.total != block.count:
                raise TableInputError(f'kappa block {index} partitions {partition.total}, but nu block size is {block.count}')
    if record.block_lambda is not None and record.block_lambda.special is None and record.nu_blocks:
        if len(record.block_lambda.values) != len(record.nu_blocks):
            raise TableInputError('Lambda must contain one block angular momentum per complete-channel block')
    if record.parent_lambda is not None and record.parent_lambda.total is not None:
        if record.parent_lambda.total != record.N:
            raise TableInputError(f'lambda partitions {record.parent_lambda.total}, but N={record.N}')
    if record.d_blocks is not None:
        if record.nu_blocks and len(record.d_blocks) != len(record.nu_blocks):
            raise TableInputError('d_blocks must contain one d_b value per complete-channel block')
        if any((value < 0 for value in record.d_blocks)):
            raise TableInputError('d_blocks values must be nonnegative')
        product = math.prod(record.d_blocks)
        if record.d_value is not None and product != record.d_value:
            raise TableInputError(f'product(d_blocks)={product}, but d_kappa_Lambda={record.d_value}')
    if record.profile_kind not in {'M', 'alpha', 'contribution'}:
        raise TableInputError('profile_kind must be M, alpha, or contribution')
    if record.count_scope not in {None, 'lambda', 'global', 'row'}:
        raise TableInputError('count_scope must be lambda, global, row, or omitted')
    computed_total = record.profile.computed_total()
    if computed_total is not None and record.profile.total_override is not None and (computed_total != record.profile.total_override):
        raise TableInputError(f'L-resolved values sum to {computed_total}, but total_mult={record.profile.total_override}')

def _auto_tags(record):
    tags = list(record.tags)
    if record.parent_lambda == _trivial_parent(record.N) and 'ACE' not in tags:
        tags.append('ACE')
    if record.parent_lambda == _sign_parent(record.N) and (not any(('sgn' in tag or 'sign' in tag.lower() for tag in tags))):
        tags.append('$\\lambda=\\mathrm{sgn}$')
    if record.kappa is not None and record.kappa.special is None:
        if record.kappa.partitions and all((part.is_one_row() for part in record.kappa.partitions)):
            if not any(('kappa' in tag and ('mathbf 1' in tag or 'triv' in tag.lower()) for tag in tags)):
                tags.append('$\\boldsymbol\\kappa=\\mathbf 1$')
        elif record.kappa.partitions and all((part.is_one_column() for part in record.kappa.partitions)):
            if not any(('kappa' in tag and ('sgn' in tag or 'sign' in tag.lower()) for tag in tags)):
                tags.append('$\\boldsymbol\\kappa=\\mathrm{sgn}$')
    return tuple(dict.fromkeys(tags))

def normalize_raw_row(raw, *, source_index, infer_trivial_labels=False, validate=True, auto_tags=True):
    eta = parse_sequence_spec(_get_alias(raw, ETA_ALIASES), field_name='eta')
    ell = parse_sequence_spec(_get_alias(raw, ELL_ALIASES), field_name='l')
    blocks = parse_nu_spec(_get_alias(raw, NU_ALIASES))
    if blocks is None and eta is not None and (ell is not None):
        blocks = derive_channel_blocks(eta, ell)
    blocks = blocks or ()
    kappa = parse_kappa(_get_alias(raw, KAPPA_ALIASES))
    block_lambda = parse_block_lambda(_get_alias(raw, BLOCK_LAMBDA_ALIASES))
    parent_lambda = parse_partition(_get_alias(raw, PARENT_LAMBDA_ALIASES), field_name='lambda')
    N = _infer_N(raw, eta, ell, blocks, parent_lambda)
    if kappa is not None and kappa.special in {'trivial', 'sign'} and blocks:
        kappa = _trivial_kappa(blocks) if kappa.special == 'trivial' else _sign_kappa(blocks)
    if parent_lambda is not None and parent_lambda.special in {'trivial', 'sign'}:
        parent_lambda = _trivial_parent(N) if parent_lambda.special == 'trivial' else _sign_parent(N)
    sector = str(_get_alias(raw, ('sector', 'representation', 'irrep_sector')) or '').strip().lower()
    tags = _normalize_tags(_get_alias(raw, ('tags', 'tag')))
    if infer_trivial_labels:
        wants_ace = sector in {'ace', 'trivial', 'symmetric', 'bosonic'} or 'ACE' in tags
        wants_sign = sector in {'sign', 'antisymmetric', 'fermionic'} or any(('sgn' in tag or 'sign' in tag.lower() for tag in tags))
        if kappa is None and blocks:
            kappa = _sign_kappa(blocks) if wants_sign else _trivial_kappa(blocks)
        if parent_lambda is None:
            if wants_sign:
                parent_lambda = _sign_parent(N)
            elif wants_ace or sector == '':
                parent_lambda = _trivial_parent(N)
        if block_lambda is None:
            block_lambda = LambdaSpec(special='all')
    c_value = _parse_optional_int(_get_alias(raw, C_ALIASES), field_name='c_kappa_lambda')
    d_value = _parse_optional_int(_get_alias(raw, D_ALIASES), field_name='d_kappa_Lambda')
    d_blocks = _parse_int_tuple(_get_alias(raw, D_BLOCKS_ALIASES), field_name='d_b')
    if d_value is None and d_blocks is not None:
        d_value = math.prod(d_blocks)
    total_override = _parse_optional_int(_get_alias(raw, TOTAL_ALIASES), field_name='profile total')
    reported_value = _parse_reported_value(_get_alias(raw, REPORTED_ALIASES))
    code_counts = _extract_code_counts(raw)
    count_scope = _normalize_count_scope(_get_alias(raw, COUNT_SCOPE_ALIASES))
    row_color_value = _get_alias(raw, ROW_COLOR_ALIASES)
    row_color = None if _is_blank(row_color_value) else str(row_color_value).strip()
    explicit_profile_kind = _get_alias(raw, PROFILE_KIND_ALIASES)
    inferred_kind = None
    points = _extract_wide_L_points(raw)
    if points:
        inferred_kind = 'contribution'
    M_name, M_mults = _get_alias_with_name(raw, M_PROFILE_ALIASES)
    alpha_name, alpha_mults = _get_alias_with_name(raw, ALPHA_PROFILE_ALIASES)
    generic_name, generic_mults = _get_alias_with_name(raw, GENERIC_PROFILE_ALIASES)
    explicit_mults = M_mults if M_mults is not None else alpha_mults if alpha_mults is not None else generic_mults
    if M_mults is not None:
        inferred_kind = 'M'
    elif alpha_mults is not None:
        inferred_kind = 'alpha'
    if explicit_mults is not None:
        parsed = parse_L_mults(explicit_mults)
        for L, mult in parsed.items():
            points[L] = points.get(L, 0) + mult
    segments = []
    scalar_name, scalar_L = _get_alias_with_name(raw, L_SCALAR_ALIASES)
    support_name, support_value = _get_alias_with_name(raw, L_SUPPORT_ALIASES)
    if support_name in {'M_ranges', 'M_Lambda_L_ranges'}:
        inferred_kind = 'M'
    elif support_name == 'alpha_ranges':
        inferred_kind = 'alpha'
    elif support_name == 'contribution_ranges':
        inferred_kind = 'contribution'
    multiplicity_kind = str(_get_alias(raw, ('multiplicity_kind', 'value_kind', 'mode')) or '').strip().lower()
    M_per_name, M_per_value = _get_alias_with_name(raw, M_PER_L_ALIASES)
    alpha_per_name, alpha_per_value = _get_alias_with_name(raw, ALPHA_PER_L_ALIASES)
    contrib_name, contrib_value = _get_alias_with_name(raw, CONTRIBUTION_PER_L_ALIASES)
    per_L_value = M_per_value if M_per_value is not None else alpha_per_value if alpha_per_value is not None else contrib_value
    if M_per_value is not None:
        inferred_kind = 'M'
    elif alpha_per_value is not None:
        inferred_kind = 'alpha'
    elif contrib_value is not None:
        inferred_kind = 'contribution'
    if scalar_name == 'L_R' and contrib_value is not None:
        inferred_kind = 'contribution'
    if points:
        segments.extend(compress_L_points(points))
    elif scalar_L is not None:
        support = parse_L_support(scalar_L)
        if len(support) != 1 or support[0].count != 1:
            if per_L_value is not None and multiplicity_kind in {'per_l', 'per-l', 'profile', 'constant', 'm'}:
                mult = _as_int(per_L_value, field_name='value per L')
                segments.extend((replace(segment, multiplicity=mult) for segment in support))
            else:
                segments.extend(support)
                if reported_value is None and per_L_value is not None:
                    reported_value = _parse_reported_value(per_L_value)
                    if 'YE3T' not in code_counts:
                        code_counts['YE3T'] = reported_value
        elif per_L_value is not None:
            segments.append(replace(support[0], multiplicity=_as_int(per_L_value, field_name='value at L')))
        else:
            segments.extend(support)
    elif support_value is not None:
        support = parse_L_support(support_value)
        if per_L_value is not None and multiplicity_kind in {'per_l', 'per-l', 'profile', 'constant', 'm'}:
            mult = _as_int(per_L_value, field_name='value per L')
            segments.extend((replace(segment, multiplicity=mult) for segment in support))
        else:
            segments.extend(support)
            if reported_value is None and per_L_value is not None:
                reported_value = _parse_reported_value(per_L_value)
                if 'YE3T' not in code_counts:
                    code_counts['YE3T'] = reported_value
    elif _get_alias(raw, ('L_min', 'L_max')) is not None:
        start_value = _get_alias(raw, ('L_min',))
        stop_value = _get_alias(raw, ('L_max',))
        step_value = _get_alias(raw, ('L_step',)) or 1
        segment = LSegment(_as_int(start_value, field_name='L_min'), _as_int(stop_value, field_name='L_max'), _as_int(step_value, field_name='L_step'))
        if per_L_value is not None and multiplicity_kind in {'per_l', 'per-l', 'profile', 'constant', 'm'}:
            segment = replace(segment, multiplicity=_as_int(per_L_value, field_name='value per L'))
        elif reported_value is None and per_L_value is not None:
            reported_value = _parse_reported_value(per_L_value)
            if 'YE3T' not in code_counts:
                code_counts['YE3T'] = reported_value
        segments.append(segment)
    profile_kind = _normalize_profile_kind(explicit_profile_kind, default=inferred_kind or 'M')
    profile = MultiplicityProfile(segments=segments, total_override=total_override, reported_value=reported_value)
    row_id_value = _get_alias(raw, ('row_id', 'group_id', 'id'))
    record = TableRecord(N=N, eta=eta, ell=ell, nu_blocks=blocks, kappa=kappa, block_lambda=block_lambda, parent_lambda=parent_lambda, profile=profile, c_value=c_value, d_value=d_value, d_blocks=d_blocks, profile_kind=profile_kind, code_counts=code_counts, count_scope=count_scope, tags=tags, row_color=row_color, row_id=None if _is_blank(row_id_value) else str(row_id_value), source_index=source_index)
    if validate:
        _validate_record(record)
    if auto_tags:
        record.tags = _auto_tags(record)
    return record

def _merge_point_records(records, *, duplicate_policy='sum'):
    """Merge scalar/per-L rows sharing the same representation metadata.

    Code-count mappings are merged non-destructively so a printed total can be
    supplied on only one of several per-L lines. Conflicting nonblank values for
    the same code are rejected.
    """
    grouped_points = OrderedDict()
    passthrough = []
    for record in records:
        can_merge = record.profile.known() and record.profile.reported_value is None and (record.profile.total_override is None)
        if not can_merge:
            passthrough.append(record)
            continue
        try:
            points = record.profile.to_points(limit=200000)
        except TableInputError:
            passthrough.append(record)
            continue
        key = record.grouping_key()
        if key not in grouped_points:
            grouped_points[key] = (record, {})
        base, accumulator = grouped_points[key]
        merged_counts = OrderedDict(base.code_counts)
        for code, value in record.code_counts.items():
            existing_key = next((key0 for key0 in merged_counts if key0.lower() == code.lower()), None)
            if existing_key is None:
                merged_counts[code] = value
            else:
                existing = merged_counts[existing_key]
                if existing is None:
                    merged_counts[existing_key] = value
                elif value is None or value == existing:
                    pass
                else:
                    raise TableInputError(f'Conflicting code count for {code!r}: {existing!r} versus {value!r}')
        if merged_counts != base.code_counts:
            base = replace(base, code_counts=merged_counts)
            grouped_points[key] = (base, accumulator)
        for L, value in points.items():
            if L in accumulator:
                if duplicate_policy == 'error':
                    raise TableInputError(f'Duplicate value for L={L} in one representation row')
                if duplicate_policy == 'last':
                    accumulator[L] = value
                else:
                    accumulator[L] += value
            else:
                accumulator[L] = value
    merged = [replace(base, profile=MultiplicityProfile(compress_L_points(points))) for base, points in grouped_points.values()]
    all_records = merged + passthrough
    return sorted(all_records, key=lambda record: record.source_index)

def _read_text(path):
    if str(path) == '-':
        return sys.stdin.read()
    return Path(path).read_text(encoding='utf-8')

def _parse_key_value_line(line):
    separators = ';' if ';' in line else '\t'
    pieces = _split_top_level(line, separators=separators)
    row = {}
    for piece in pieces:
        if '=' not in piece:
            raise TableInputError(f'Printed line item {piece!r} is not key=value; separate fields by semicolons')
        key, value = piece.split('=', 1)
        row[key.strip()] = value.strip()
    return row

def _load_json_text(text):
    data = json.loads(text)
    if isinstance(data, Mapping):
        if 'rows' in data:
            data = data['rows']
        else:
            data = [data]
    if not isinstance(data, list):
        raise TableInputError("JSON input must be an object, an array, or {'rows': [...]} ")
    return [dict(item) for item in data]

def _load_jsonl_text(text):
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise TableInputError(f'Invalid JSON on line {line_number}') from exc
        if not isinstance(item, Mapping):
            raise TableInputError(f'JSONL line {line_number} must be an object')
        rows.append(dict(item))
    return rows

def _load_python_literal_text(text):
    try:
        data = ast.literal_eval(text)
    except Exception as exc:
        raise TableInputError('Input is not a safe Python literal') from exc
    if isinstance(data, Mapping):
        data = [data]
    if not isinstance(data, (list, tuple)) or not all((isinstance(item, Mapping) for item in data)):
        raise TableInputError('Python literal input must be a dict or a list of dicts')
    return [dict(item) for item in data]

def _load_key_value_text(text):
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        try:
            rows.append(_parse_key_value_line(stripped))
        except TableInputError as exc:
            raise TableInputError(f'Line {line_number}: {exc}') from exc
    return rows

def load_raw_rows(path, *, input_format='auto'):
    path_text = str(path)
    suffix = Path(path_text).suffix.lower() if path_text != '-' else ''
    fmt = input_format.lower()
    if fmt == 'auto':
        fmt = {'.csv': 'csv', '.json': 'json', '.jsonl': 'jsonl', '.ndjson': 'jsonl', '.py': 'python'}.get(suffix, 'text')
    if fmt == 'csv':
        if path_text == '-':
            reader = csv.DictReader(sys.stdin)
            return [dict(row) for row in reader]
        with Path(path).open('r', encoding='utf-8', newline='') as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    text = _read_text(path)
    if fmt == 'json':
        return _load_json_text(text)
    if fmt == 'jsonl':
        return _load_jsonl_text(text)
    if fmt == 'python':
        return _load_python_literal_text(text)
    if fmt in {'kv', 'keyvalue', 'printed'}:
        return _load_key_value_text(text)
    if fmt != 'text':
        raise TableInputError(f'Unsupported input format {input_format!r}')
    stripped = text.lstrip()
    if stripped.startswith('{') or stripped.startswith('['):
        try:
            return _load_json_text(text)
        except Exception:
            try:
                return _load_python_literal_text(text)
            except Exception:
                pass
    try:
        rows = _load_jsonl_text(text)
        if rows:
            return rows
    except Exception:
        pass
    return _load_key_value_text(text)

def load_records(path, *, input_format='auto', infer_trivial_labels=False, validate=True, auto_tags=True, duplicate_policy='sum', defaults=None):
    raw_rows = load_raw_rows(path, input_format=input_format)
    defaults = dict(defaults or {})
    records = []
    errors = []
    for index, raw in enumerate(raw_rows, start=1):
        try:
            merged_raw = dict(defaults)
            merged_raw.update(raw)
            records.append(normalize_raw_row(merged_raw, source_index=index, infer_trivial_labels=infer_trivial_labels, validate=validate, auto_tags=auto_tags))
        except TableInputError as exc:
            errors.append(f'row {index}: {exc}')
    if errors:
        raise TableInputError('\n'.join(errors))
    return _merge_point_records(records, duplicate_policy=duplicate_policy)

def write_normalized_jsonl(records, path):
    with Path(path).open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record.to_jsonable(), ensure_ascii=False, sort_keys=True) + '\n')

@dataclass(frozen=True)
class FormatOptions:
    __field_names__ = (
        'partition_style', 'kappa_style', 'lambda_style', 'diagram_max_rank',
        'ydiagram_max_cells', 'ydiagram_max_rows', 'max_channel_blocks',
        'rank_placement', 'group_placement', 'show_nu', 'show_kappa',
        'show_block_lambda', 'show_parent_lambda', 'show_tags', 'row_colors',
        'ace_row_color', 'sign_row_color', 'mixed_row_color', 'show_cd',
        'show_d_blocks', 'show_input_channels', 'count_codes', 'count_scope',
        'count_symbol', 'profile_max_items', 'profile_columns',
        'prune_empty_count_codes', 'sequence_power_min_rank', 'split_by_rank',
        'group_resolved_rows'
    )
    partition_style = 'auto'
    kappa_style = None
    lambda_style = None
    diagram_max_rank = 8
    ydiagram_max_cells = 256
    ydiagram_max_rows = 256
    max_channel_blocks = 8
    rank_placement = 'column'
    group_placement = 'column'
    show_nu = True
    show_kappa = True
    show_block_lambda = True
    show_parent_lambda = True
    show_tags = False
    row_colors = 'auto'
    ace_row_color = 'blue!8'
    sign_row_color = 'green!10'
    mixed_row_color = ''
    show_cd = 'auto'
    show_d_blocks = False
    show_input_channels = False
    count_codes = ('YE3T',)
    count_scope = 'auto'
    count_symbol = None
    profile_max_items = 10
    profile_columns = 'omit'
    prune_empty_count_codes = True
    sequence_power_min_rank = 5
    split_by_rank = False
    group_resolved_rows = False

def _normalized_partition_style(options):
    return 'partition' if options.partition_style == 'tuple' else options.partition_style

def _with_partition_style(options, style):
    """Return options with a per-column partition-style override applied."""
    return options if style is None else replace(options, partition_style=style)

def format_partition(partition, options):
    if partition is None:
        return '\\mathrm{n/r}'
    if partition.special is not None:
        if partition.special.startswith('latex:'):
            return partition.special.split(':', 1)[1]
        if partition.special == 'all':
            return '\\sum_{\\lambda}'
        if partition.special == 'trivial':
            return '\\mathbf 1'
        if partition.special == 'sign':
            return '\\mathrm{sgn}'
        return '\\mathrm{n/r}'
    style = _normalized_partition_style(options)
    use_ydiagram = style == 'ydiagram'
    if style == 'auto':
        use_ydiagram = (partition.total or 0) <= options.diagram_max_rank and (partition.total or 0) <= options.ydiagram_max_cells and ((partition.number_of_rows or 0) <= options.ydiagram_max_rows)
    if use_ydiagram:
        if (partition.total or 0) > options.ydiagram_max_cells:
            raise TableInputError('Forced ydiagram exceeds --ydiagram-max-cells; use partition/compact notation or explicitly raise the safety limit.')
        if (partition.number_of_rows or 0) > options.ydiagram_max_rows:
            raise TableInputError('Forced ydiagram exceeds --ydiagram-max-rows; use partition/compact notation.')
        parts = partition.expanded(max_items=options.ydiagram_max_rows)
        return '\\ydiagram{' + ','.join(map(str, parts)) + '}'
    terms = [str(part) if count == 1 else f'{part}^{{{count}}}' for part, count in partition.runs]
    body = ','.join(terms)
    if style == 'compact':
        return '(' + body + ')'
    return '\\left(' + body + '\\right)'

def format_kappa(kappa, options):
    if kappa is None:
        return '\\mathrm{n/r}'
    if kappa.special is not None:
        if kappa.special == 'all':
            return '\\sum_{\\boldsymbol\\kappa}'
        if kappa.special == 'trivial':
            return '\\mathbf 1_{G_{\\boldsymbol\\nu}}'
        if kappa.special == 'sign':
            return '\\mathrm{sgn}_{G_{\\boldsymbol\\nu}}'
        return '\\mathrm{n/r}'
    local_options = _with_partition_style(options, options.kappa_style)
    style = _normalized_partition_style(local_options)
    if style == 'auto':
        all_small = all((part.total is not None and part.total <= local_options.diagram_max_rank and ((part.total or 0) <= local_options.ydiagram_max_cells) and ((part.number_of_rows or 0) <= local_options.ydiagram_max_rows) for part in kappa.partitions))
        if not all_small:
            local_options = replace(local_options, partition_style='compact')
    body = '\\mid'.join((format_partition(part, local_options) for part in kappa.partitions))
    if _normalized_partition_style(local_options) == 'compact':
        return body
    return '\\left(' + body + '\\right)'

def format_block_lambda(block_lambda):
    if block_lambda is None:
        return '\\mathrm{n/r}'
    if block_lambda.special is not None:
        return '\\sum_{\\boldsymbol\\Lambda}' if block_lambda.special == 'all' else '\\mathrm{n/r}'
    return '\\left(' + ','.join(map(str, block_lambda.values)) + '\\right)'

def format_d_blocks(values):
    if values is None:
        return '--'
    return '\\left(' + ','.join(map(str, values)) + '\\right)'

def format_sequence(seq):
    if seq is None:
        return '\\text{not supplied}'
    pieces = []
    for value, count in seq.runs:
        atom = _latex_atom(value)
        if count == 1:
            pieces.append('[' + atom + ']')
        else:
            pieces.append('[' + atom + f']^{{\\times {count}}}')
    return '+'.join(pieces) if pieces else '[]'

def format_plain_sequence_from_blocks(blocks, attr, options):
    values = []
    for block in blocks:
        value = block.eta if attr == 'eta' else block.ell
        values.extend([value] * int(block.count))
    if not values:
        return '\\text{not supplied}'
    use_powers = len(values) >= int(options.sequence_power_min_rank)
    if use_powers:
        pieces = []
        current = values[0]
        count = 0
        for value in values:
            if value == current:
                count += 1
                continue
            atom = _latex_atom(current)
            pieces.append(atom if count == 1 else f'{atom}^{{{count}}}')
            current = value
            count = 1
        atom = _latex_atom(current)
        pieces.append(atom if count == 1 else f'{atom}^{{{count}}}')
        return '\\left(' + ','.join(pieces) + '\\right)'
    return '\\left(' + ','.join((_latex_atom(value) for value in values)) + '\\right)'

def format_nu(blocks, options):
    if not blocks:
        return '\\text{not supplied}'
    if len(blocks) > options.max_channel_blocks:
        return f'\\left\\{{(\\eta_b,l_b)^{{k_b}}\\right\\}}_{{b=1}}^{{{len(blocks)}}}'
    pieces = []
    for block in blocks:
        pair = '\\left(' + _latex_atom(block.eta) + ',' + _latex_atom(block.ell) + '\\right)'
        if block.count != 1:
            pair += f'^{{{block.count}}}'
        pieces.append(pair)
    return '\\left(' + ',\\,'.join(pieces) + '\\right)'

def format_group(blocks, options):
    if not blocks:
        return '\\text{not supplied}'
    if len(blocks) > options.max_channel_blocks:
        return f'\\prod_{{b=1}}^{{{len(blocks)}}}S_{{k_b}}'
    factors = [f'S_{{{block.count}}}' for block in blocks]
    return '\\,'.join(factors)

def format_L_segment(segment):
    if segment.start == segment.stop:
        return f'L={segment.start}'
    if segment.count == 2:
        return f'L\\in\\{{{segment.start},{segment.stop}\\}}'
    if segment.step == 1:
        return f'{segment.start}\\le L\\le {segment.stop}'
    if segment.start == 0:
        return f'L=0,{segment.step},\\ldots,{segment.stop}'
    return f'L={segment.start},{segment.start + segment.step},\\ldots,{segment.stop}'

def format_L_range_values(segment):
    if segment.start == segment.stop:
        return str(segment.start)
    if segment.count == 2:
        return f'\\{{{segment.start},{segment.stop}\\}}'
    if segment.step == 1:
        return f'{segment.start}\\!:\\!{segment.stop}'
    return f'{segment.start},{segment.start + segment.step},\\ldots,{segment.stop}'

def _merge_support_segments(segments):
    values_count = sum((segment.count for segment in segments))
    if values_count <= 200000:
        values = set()
        for segment in segments:
            values.update(segment.iter_values())
        return compress_L_support(values)
    return [replace(segment, multiplicity=None) for segment in segments]

def format_L_support(profile, *, max_items=8):
    segments = _merge_support_segments(profile.support_segments())
    if not segments:
        return '--'
    shown = segments[:max_items]
    body = ';\\;'.join((format_L_segment(segment) for segment in shown))
    if len(segments) > max_items:
        body += ';\\;\\ldots'
    return body

def format_simple_L_bounds(profile):
    segments = profile.support_segments()
    if not segments:
        return '--'
    starts = [segment.start for segment in segments]
    stops = [segment.stop for segment in segments]
    lo = min(starts)
    hi = max(stops)
    if lo == hi:
        return f'L={lo}'
    return f'{lo}\\le L\\le {hi}'

def format_allowed_lambda_header(records, options):
    if records and all((record.parent_lambda is not None and record.parent_lambda.is_one_row() and (record.parent_lambda.total == record.N) for record in records)):
        return '$\\lambda=(N)$'
    return '$\\lambda,L$'

def format_allowed_lambda_L(record, options):
    if record.parent_lambda is not None and record.parent_lambda.is_one_row() and (record.parent_lambda.total == record.N):
        return format_simple_L_bounds(record.profile)
    lambda_options = _with_partition_style(options, options.lambda_style)
    lam = format_partition(record.parent_lambda, lambda_options)
    return f'{lam}\\,\\vert\\,{format_simple_L_bounds(record.profile)}'

def _profile_symbol_for_kind(kind):
    if kind == 'M':
        return 'M_{\\boldsymbol\\Lambda}^{L}'
    if kind == 'alpha':
        return '\\alpha_{\\boldsymbol{\\nu}^{\\circ}}^{\\lambda L}'
    return 'c_{\\boldsymbol\\kappa}^{\\lambda}d_{\\boldsymbol\\kappa\\boldsymbol\\Lambda}M_{\\boldsymbol\\Lambda}^{L}'

def _profile_symbol(records):
    kinds = {record.profile_kind for record in records}
    if len(kinds) == 1:
        return _profile_symbol_for_kind(next(iter(kinds)))
    return '\\text{$L$-resolved value}'

def format_profile(record, *, max_items=10):
    profile = record.profile
    if not profile.segments:
        return '--'
    if not profile.known():
        return '\\text{aggregate only}'
    symbol = _profile_symbol_for_kind(record.profile_kind)
    entries = []
    for segment in profile.segments[:max_items]:
        assert segment.multiplicity is not None
        if segment.start == segment.stop:
            entries.append(f'{symbol}\\big|_{{L={segment.start}}}={segment.multiplicity}')
        else:
            entries.append(f'{symbol}={segment.multiplicity}\\;({format_L_segment(segment)})')
    if len(profile.segments) > max_items:
        entries.append('\\ldots')
    return ',\\;'.join(entries)

def format_value(value):
    if value is None:
        return '--'
    if isinstance(value, int):
        return str(value)
    text = value.strip()
    if text.startswith('$') and text.endswith('$'):
        return text[1:-1]
    return '\\text{' + _latex_escape_text(text) + '}'

def format_tags(tags):
    if not tags:
        return ''
    rendered = []
    for tag in tags:
        stripped = tag.strip()
        if stripped.startswith('$') and stripped.endswith('$'):
            rendered.append(stripped[1:-1])
        elif '\\' in stripped:
            rendered.append(stripped)
        else:
            rendered.append('\\text{' + _latex_escape_text(stripped) + '}')
    return ';\\;'.join(rendered)

def _show_cd(records, options):
    if options.show_cd == 'yes':
        return (True, True)
    if options.show_cd == 'no':
        return (False, False)
    return (any((record.c_value is not None for record in records)), any((record.d_value is not None for record in records)))

def _table_parts(rows, max_rows):
    if max_rows <= 0:
        raise ValueError('max_rows must be positive')
    if not rows:
        return [[]]
    return [rows[start:start + max_rows] for start in range(0, len(rows), max_rows)]

def _caption_for_part(caption, part_index, n_parts):
    return caption if n_parts == 1 else caption + f' (part {part_index} of {n_parts}).'

def _label_for_part(label, part_index, n_parts):
    return label if n_parts == 1 else f'{label}_part_{part_index}'

def _uses_ydiagram(records, options):
    style = _normalized_partition_style(options)
    if style == 'ydiagram':
        return True
    if style != 'auto':
        return False
    partitions = []
    for record in records:
        if options.show_parent_lambda and record.parent_lambda is not None and (record.parent_lambda.special is None):
            partitions.append(record.parent_lambda)
        if options.show_kappa and record.kappa is not None and (record.kappa.special is None):
            partitions.extend(record.kappa.partitions)
    return any((part.total is not None and part.total <= options.diagram_max_rank for part in partitions))

def _common_header_comment(records, options):
    comments = [f'% Generated by ye3t.multiplicity_tables v{__version__}', '% Required packages: amsmath, booktabs, graphicx']
    if _uses_ydiagram(records, options):
        comments.append('% Also load ytableau for \\ydiagram output.')
    if options.group_resolved_rows:
        comments.append('% Also load multirow for grouped resolved-row tables.')
    if options.row_colors != 'off':
        comments.append('% Row coloring uses \\rowcolor; load \\usepackage[table]{xcolor}.')
    comments.extend(['% Output notation uses final angular momentum L.', '% d_{kappa Lambda}=product_b d_b^{kappa_b Lambda_b}.', '% Pure recoupling profiles are denoted M_{Lambda}^{L}.', ''])
    return comments

def _record_sort_key(record):
    tag_priority = 0 if 'ACE' in record.tags else 1 if any(('sgn' in tag or 'sign' in tag.lower() for tag in record.tags)) else 2
    return (record.N, tuple((block.key() for block in record.nu_blocks)), tag_priority, str(record.kappa), str(record.block_lambda), str(record.parent_lambda), record.source_index)

def _rank_groups(records, options):
    if not options.split_by_rank:
        return [(None, list(records))]
    grouped = OrderedDict()
    for record in records:
        grouped.setdefault(record.N, []).append(record)
    return list(grouped.items())

def _row_color_command(record, options):
    if options.row_colors == 'off':
        return None
    if record.row_color:
        return f'\\rowcolor{{{record.row_color}}}'
    if 'ACE' in record.tags and options.ace_row_color:
        return f'\\rowcolor{{{options.ace_row_color}}}'
    if any(('sgn' in tag or 'sign' in tag.lower() for tag in record.tags)) and options.sign_row_color:
        return f'\\rowcolor{{{options.sign_row_color}}}'
    if options.mixed_row_color:
        return f'\\rowcolor{{{options.mixed_row_color}}}'
    return None

def _requested_count_codes(records, options):
    requested = tuple((code.strip() for code in options.count_codes if code.strip()))
    if not requested:
        return ()
    if len(requested) == 1 and requested[0].lower() == 'auto':
        available = []
        for preferred in KNOWN_COUNT_CODES:
            if any((any((key.lower() == preferred.lower() for key in record.code_counts)) for record in records)):
                available.append(preferred)
        for record in records:
            for code in record.code_counts:
                canonical = _canonical_count_code(code)
                if canonical not in available:
                    available.append(canonical)
        if not available:
            available.append('YE3T')
        return tuple(available)
    if len(requested) == 1 and requested[0].lower() in {'none', 'off'}:
        return ()
    codes = tuple((_canonical_count_code(code) for code in requested))
    if not options.prune_empty_count_codes:
        return codes
    return tuple((code for code in codes if any((_count_value_populated(_lookup_count(record, code)) for record in records))))

def _lookup_count(record, code):
    for key, value in record.code_counts.items():
        if key.lower() == code.lower():
            return value
    if code.lower() == 'ye3t':
        return record.derived_row_count()
    return None

def _count_value_populated(value):
    if isinstance(value, int):
        return True
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {'', '--', 'not run', 'rank_skip', 'rank skip', 'error', 'none', 'null'}

def format_count_value(value):
    if isinstance(value, int):
        return str(value)
    if value is None:
        return ''
    text = str(value).strip()
    lower = text.lower()
    if 'timeout' in lower:
        return '--'
    if lower in {'', '--', 'not run', 'rank_skip', 'rank skip', 'error', 'none', 'null'}:
        return ''
    return format_value(text)

def _resolved_count_scope(records, options, layout):
    if options.count_scope != 'auto':
        return options.count_scope
    explicit = {record.count_scope for record in records if record.count_scope is not None}
    if len(explicit) == 1:
        return next(iter(explicit))
    if len(explicit) > 1:
        return 'row'
    if all((record.parent_lambda is not None and record.parent_lambda.special == 'all' for record in records)):
        return 'global'
    if all((record.code_counts and record.parent_lambda is not None and (record.parent_lambda.special not in {'all', 'unresolved'}) for record in records)):
        return 'lambda'
    if layout == 'summary' and all((record.parent_lambda is not None and record.parent_lambda.special not in {'all', 'unresolved'} and (record.kappa is None or record.kappa.special in {'all', 'unresolved'}) and (record.block_lambda is None or record.block_lambda.special in {'all', 'unresolved'}) for record in records)):
        return 'lambda'
    return 'row'

def _count_header(records, options, layout):
    if options.count_symbol:
        return options.count_symbol
    scope = _resolved_count_scope(records, options, layout)
    if scope == 'global':
        return '\\mathcal{D}_{\\boldsymbol{\\nu}^{\\circ}}'
    if scope == 'lambda':
        return '\\mathcal{D}_{\\boldsymbol{\\nu}^{\\circ}}^{\\lambda}'
    return '\\sum_L c_{\\boldsymbol\\kappa}^{\\lambda}d_{\\boldsymbol\\kappa\\boldsymbol\\Lambda}M_{\\boldsymbol\\Lambda}^{L}'

def _context_group_key(record, options):
    key = []
    if options.rank_placement == 'header':
        key.append(record.N)
    if options.group_placement == 'header':
        key.append(tuple((block.key() for block in record.nu_blocks)))
    return tuple(key)

def _context_groups(records, options):
    if options.rank_placement != 'header' and options.group_placement != 'header':
        return [list(records)]
    grouped = OrderedDict()
    for record in records:
        grouped.setdefault(_context_group_key(record, options), []).append(record)
    return list(grouped.values())

def _context_header(records, options, ncols):
    pieces = []
    if options.rank_placement == 'header':
        ranks = {record.N for record in records}
        if len(ranks) != 1:
            raise TableInputError('rank-placement=header requires a single N in each output table')
        pieces.append(f'N={next(iter(ranks))}')
    if options.group_placement == 'header':
        groups = {tuple((block.key() for block in record.nu_blocks)) for record in records}
        if len(groups) != 1:
            raise TableInputError('group-placement=header requires a single G_nu in each output table')
        pieces.append(f'G_{{\\boldsymbol\\nu}}={format_group(records[0].nu_blocks, options)}')
    if not pieces:
        return None
    body = '\\qquad '.join(pieces)
    return f'\\multicolumn{{{ncols}}}{{l}}{{${body}$}} \\\\'

def _base_headers_and_colspec(records, options):
    show_c, show_d = _show_cd(records, options)
    headers = []
    second_headers = []
    colspec = []
    if options.rank_placement == 'column':
        headers.append('$N$')
        second_headers.append('')
        colspec.append('c')
    if options.show_input_channels:
        headers.extend(['$\\boldsymbol\\eta_{\\rm in}$', '$\\boldsymbol l_{\\rm in}$'])
        second_headers.extend(['', ''])
        colspec.extend(['p{2.0cm}', 'p{2.0cm}'])
    if options.show_nu:
        headers.append('\\multicolumn{2}{c}{$\\boldsymbol{\\nu}^{\\circ}$}')
        second_headers.extend(['$\\boldsymbol{\\eta}$', '$\\boldsymbol{l}$'])
        colspec.extend(['p{1.55cm}', 'p{1.55cm}'])
    if options.group_placement == 'column':
        headers.append('$G_{\\boldsymbol\\nu}$')
        second_headers.append('')
        colspec.append('p{1.8cm}')
    if options.show_kappa:
        headers.append('$\\boldsymbol\\kappa$')
        second_headers.append('')
        colspec.append('p{2.35cm}')
    if options.show_block_lambda:
        headers.append('$\\boldsymbol\\Lambda$')
        second_headers.append('')
        colspec.append('p{1.35cm}')
    if options.show_parent_lambda:
        headers.append('$\\lambda$')
        second_headers.append('')
        colspec.append('p{1.25cm}')
    if show_c:
        headers.append('$c_{\\boldsymbol\\kappa}^{\\lambda}$')
        second_headers.append('')
        colspec.append('c')
    if show_d:
        headers.append('$d_{\\boldsymbol\\kappa\\boldsymbol\\Lambda}$')
        second_headers.append('')
        colspec.append('c')
    if options.show_d_blocks:
        headers.append('$\\{d_b^{\\kappa_b\\Lambda_b}\\}_{b=1}^{B}$')
        second_headers.append('')
        colspec.append('p{2.0cm}')
    return (headers, second_headers, colspec, show_c, show_d)

def _descriptor_cells(record, options):
    cells = []
    if options.rank_placement == 'column':
        cells.append(f'${record.N}$')
    if options.show_input_channels:
        cells.extend([f'${format_sequence(record.eta)}$', f'${format_sequence(record.ell)}$'])
    if options.show_nu:
        cells.extend([f"${format_plain_sequence_from_blocks(record.nu_blocks, 'eta', options)}$", f"${format_plain_sequence_from_blocks(record.nu_blocks, 'ell', options)}$"])
    if options.group_placement == 'column':
        cells.append(f'${format_group(record.nu_blocks, options)}$')
    return cells

def _intermediate_cells(record, options, show_c, show_d):
    cells = []
    if options.show_kappa:
        cells.append(f'${format_kappa(record.kappa, options)}$')
    if options.show_block_lambda:
        cells.append(f'${format_block_lambda(record.block_lambda)}$')
    if options.show_parent_lambda:
        lambda_options = _with_partition_style(options, options.lambda_style)
        cells.append(f'${format_partition(record.parent_lambda, lambda_options)}$')
    if show_c:
        cells.append(f'${format_value(record.c_value)}$')
    if show_d:
        cells.append(f'${format_value(record.d_value)}$')
    if options.show_d_blocks:
        cells.append(f'${format_d_blocks(record.d_blocks)}$')
    return cells

def _base_cells(record, options, show_c, show_d):
    return _descriptor_cells(record, options) + _intermediate_cells(record, options, show_c, show_d)

def _append_count_header(first_headers, second_headers, colspec, records, options, layout):
    codes = _requested_count_codes(records, options)
    if not codes:
        return ('', codes, None)
    start_col = len(colspec) + 1
    first_headers.append(f'\\multicolumn{{{len(codes)}}}{{c}}{{${_count_header(records, options, layout)}$}}')
    second_headers.extend([f'\\text{{{_latex_escape_text(code)}}}' for code in codes])
    colspec.extend(['c'] * len(codes))
    return ('', codes, (start_col, start_col + len(codes) - 1))

def _append_count_cells(cells, record, codes):
    for code in codes:
        value = format_count_value(_lookup_count(record, code))
        cells.append(f'${value}$' if value else '')

def _descriptor_group_key(record):
    return (record.N, record.eta, record.ell, tuple((block.key() for block in record.nu_blocks)))

def _descriptor_row_groups(records):
    grouped = OrderedDict()
    for record in records:
        grouped.setdefault(_descriptor_group_key(record), []).append(record)
    return list(grouped.values())

def _table_unit_parts(units, max_rows):
    if max_rows <= 0:
        raise ValueError('max_rows must be positive')
    parts = []
    current = []
    current_rows = 0
    for unit in units:
        unit_rows = max(1, len(unit))
        if current and current_rows + unit_rows > max_rows:
            parts.append(current)
            current = []
            current_rows = 0
        current.append(unit)
        current_rows += unit_rows
    if current:
        parts.append(current)
    return parts or [[]]

def _flatten_units(units):
    return [record for unit in units for record in unit]

def _multirow_cell(cell, nrows):
    if nrows <= 1:
        return cell
    return f'\\multirow{{{nrows}}}{{*}}{{{cell}}}'

def _compact_tail_cells(record, options, codes, profile_L_values):
    cells = [f'${format_allowed_lambda_L(record, options)}$']
    if options.profile_columns == 'compact':
        cells.append('\\parbox[t]{\\linewidth}{\\raggedright $' + format_profile(record, max_items=options.profile_max_items) + '$}')
    elif options.profile_columns == 'subcolumns' and profile_L_values:
        points = record.profile.to_points(limit=max(100000, len(profile_L_values) * 2))
        cells.extend((str(points.get(L, 0)) for L in profile_L_values))
    _append_count_cells(cells, record, codes)
    if options.show_tags:
        cells.append(f'${format_tags(record.tags)}$' if record.tags else '')
    return cells

def _emit_compact_record_row(parts, record, options, show_c, show_d, codes, profile_L_values):
    color = _row_color_command(record, options)
    if color:
        parts.append(color)
    cells = _base_cells(record, options, show_c, show_d)
    cells.extend(_compact_tail_cells(record, options, codes, profile_L_values))
    parts.append(' & '.join(cells) + ' \\\\')

def _emit_compact_grouped_rows(parts, units, options, show_c, show_d, codes, profile_L_values):
    for unit_index, unit in enumerate(units):
        nrows = len(unit)
        for row_index, record in enumerate(unit):
            color = _row_color_command(record, options)
            if color:
                parts.append(color)
            if row_index == 0:
                cells = [_multirow_cell(cell, nrows) for cell in _descriptor_cells(record, options)]
            else:
                cells = [''] * len(_descriptor_cells(record, options))
            cells.extend(_intermediate_cells(record, options, show_c, show_d))
            cells.extend(_compact_tail_cells(record, options, codes, profile_L_values))
            parts.append(' & '.join(cells) + ' \\\\')
        if unit_index != len(units) - 1:
            parts.append('\\addlinespace[1pt]')

def _profile_L_values(records):
    values = set()
    for record in records:
        if record.profile.known():
            points = record.profile.to_points(limit=100000)
            values.update(points.keys())
    return sorted(values)

def _emit_table_preamble(parts, *, caption, label, colspec, size, tabcolsep, arraystretch, resizebox):
    parts.extend(['\\begin{table*}[t]', '\\centering', f'\\{size}', f'\\setlength{{\\tabcolsep}}{{{tabcolsep}}}', f'\\renewcommand{{\\arraystretch}}{{{arraystretch}}}', '\\caption{' + caption + '}', '\\label{' + label + '}'])
    if resizebox:
        parts.append('\\resizebox{\\textwidth}{!}{%')
    parts.extend(['\\begin{tabular}{' + colspec + '}', '\\toprule'])

def _emit_table_end(parts, *, resizebox):
    parts.extend(['\\bottomrule', '\\end{tabular}'])
    if resizebox:
        parts.append('}')
    parts.extend(['\\end{table*}', ''])

def _emit_two_level_header(parts, first, second, cmidrules, context):
    if context:
        parts.extend([context, '\\midrule'])
    parts.append(' & '.join(first) + ' \\\\')
    inferred_rules = []
    column = 1
    for header in first:
        match = re.match('\\\\multicolumn\\{(\\d+)\\}', header)
        span = int(match.group(1)) if match else 1
        if span > 1:
            inferred_rules.append((column, column + span - 1))
        column += span
    all_rules = []
    for rule in inferred_rules + list(cmidrules):
        if rule not in all_rules:
            all_rules.append(rule)
    for start, stop in all_rules:
        parts.append(f'\\cmidrule(lr){{{start}-{stop}}}')
    parts.append(' & '.join(second) + ' \\\\')
    parts.append('\\midrule')

def write_summary_table(records, output, *, caption, label, max_rows_per_table=24, options=FormatOptions(), resizebox=True):
    rows = sorted(records, key=_record_sort_key)
    context_groups = _context_groups(rows, options)
    parts = _common_header_comment(rows, options)
    for group_index, group_rows in enumerate(context_groups, start=1):
        chunks = _table_parts(group_rows, max_rows_per_table)
        for part_index, chunk in enumerate(chunks, start=1):
            headers, second, colspec, show_c, show_d = _base_headers_and_colspec(chunk, options)
            headers.append('allowed $\\lambda,L$')
            second.append(format_allowed_lambda_header(chunk, options))
            colspec.append('p{2.35cm}')
            _, codes, count_rule = _append_count_header(headers, second, colspec, chunk, options, 'summary')
            if options.show_tags:
                headers.append('tag')
                second.append('')
                colspec.append('p{1.9cm}')
            ncols = len(colspec)
            table_caption = caption
            if len(context_groups) > 1:
                table_caption += f' Context group {group_index} of {len(context_groups)}.'
            table_caption = _caption_for_part(table_caption, part_index, len(chunks))
            table_label = _sanitize_label(label)
            if len(context_groups) > 1:
                table_label += f'_context_{group_index}'
            table_label = _label_for_part(table_label, part_index, len(chunks))
            _emit_table_preamble(parts, caption=table_caption, label=table_label, colspec=''.join(colspec), size='scriptsize', tabcolsep='2pt', arraystretch='1.12', resizebox=resizebox)
            context = _context_header(chunk, options, ncols)
            _emit_two_level_header(parts, headers, second, [count_rule] if count_rule else [], context)
            for record in chunk:
                color = _row_color_command(record, options)
                if color:
                    parts.append(color)
                cells = _base_cells(record, options, show_c, show_d)
                cells.append(f'${format_allowed_lambda_L(record, options)}$')
                _append_count_cells(cells, record, codes)
                if options.show_tags:
                    cells.append(f'${format_tags(record.tags)}$' if record.tags else '')
                parts.append(' & '.join(cells) + ' \\\\')
            _emit_table_end(parts, resizebox=resizebox)
    Path(output).write_text('\n'.join(parts), encoding='utf-8')

def write_compact_table(records, output, *, caption, label, max_rows_per_table=24, options=FormatOptions(), resizebox=True):
    rows = sorted(records, key=_record_sort_key)
    parts = _common_header_comment(rows, options)
    rank_groups = _rank_groups(rows, options)
    for _, (rank_value, rank_rows) in enumerate(rank_groups, start=1):
        context_groups = _context_groups(rank_rows, options)
        for group_index, group_rows in enumerate(context_groups, start=1):
            if options.group_resolved_rows:
                row_units = _descriptor_row_groups(group_rows)
                chunks = _table_unit_parts(row_units, max_rows_per_table)
            else:
                chunks = [[(record,) for record in chunk] for chunk in _table_parts(group_rows, max_rows_per_table)]
            for part_index, chunk_units in enumerate(chunks, start=1):
                chunk = _flatten_units(chunk_units)
                headers, second, colspec, show_c, show_d = _base_headers_and_colspec(chunk, options)
                headers.append('allowed $\\lambda,L$')
                second.append(format_allowed_lambda_header(chunk, options))
                colspec.append('p{2.1cm}')
                profile_L_values = []
                profile_rule = None
                if options.profile_columns == 'compact':
                    if all((record.profile_kind == 'alpha' for record in chunk)):
                        headers.append('$\\alpha^{\\lambda L}$')
                    else:
                        headers.append(f'${_profile_symbol(chunk)}$')
                    second.append('')
                    colspec.append('p{3.2cm}')
                elif options.profile_columns == 'subcolumns':
                    profile_L_values = _profile_L_values(chunk)
                    if profile_L_values:
                        start_col = len(colspec) + 1
                        headers.append(f'\\multicolumn{{{len(profile_L_values)}}}{{c}}{{${_profile_symbol(chunk)}$}}')
                        second.extend([f'$L={L}$' for L in profile_L_values])
                        colspec.extend(['c'] * len(profile_L_values))
                        profile_rule = (start_col, start_col + len(profile_L_values) - 1)
                _, codes, count_rule = _append_count_header(headers, second, colspec, chunk, options, 'compact')
                if options.show_tags:
                    headers.append('tag')
                    second.append('')
                    colspec.append('p{1.8cm}')
                ncols = len(colspec)
                table_caption = caption
                if rank_value is not None:
                    table_caption += f' Rank {rank_value}.'
                if len(context_groups) > 1:
                    table_caption += f' Context group {group_index} of {len(context_groups)}.'
                table_caption = _caption_for_part(table_caption, part_index, len(chunks))
                table_label = _sanitize_label(label)
                if rank_value is not None:
                    table_label += f'_N_{rank_value}'
                if len(context_groups) > 1:
                    table_label += f'_context_{group_index}'
                table_label = _label_for_part(table_label, part_index, len(chunks))
                _emit_table_preamble(parts, caption=table_caption, label=table_label, colspec=''.join(colspec), size='scriptsize', tabcolsep='2pt', arraystretch='1.12', resizebox=resizebox)
                context = _context_header(chunk, options, ncols)
                rules = ([profile_rule] if profile_rule else []) + ([count_rule] if count_rule else [])
                _emit_two_level_header(parts, headers, second, rules, context)
                if options.group_resolved_rows:
                    _emit_compact_grouped_rows(parts, chunk_units, options, show_c, show_d, codes, profile_L_values)
                else:
                    for record in chunk:
                        _emit_compact_record_row(parts, record, options, show_c, show_d, codes, profile_L_values)
                _emit_table_end(parts, resizebox=resizebox)
    Path(output).write_text('\n'.join(parts), encoding='utf-8')

@dataclass(frozen=True)
class RangeRow:
    __field_names__ = ('record', 'segment', 'displayed_value')
    pass
    pass
    pass

def _range_rows(records):
    rows = []
    for record in records:
        if record.profile.segments:
            for segment in record.profile.segments:
                rows.append(RangeRow(record, segment, segment.multiplicity))
        else:
            rows.append(RangeRow(record, None, None))
    rows.sort(key=lambda row: _record_sort_key(row.record) + (-1 if row.segment is None else row.segment.start, -1 if row.segment is None else row.segment.stop, 1 if row.segment is None else row.segment.step))
    return rows

def write_range_table(records, output, *, caption, label, max_rows_per_table=28, options=FormatOptions(), resizebox=True):
    sorted_records = sorted(records, key=_record_sort_key)
    context_groups = _context_groups(sorted_records, options)
    parts = _common_header_comment(records, options)
    for group_index, group_records in enumerate(context_groups, start=1):
        range_rows = _range_rows(group_records)
        chunks = _table_parts(range_rows, max_rows_per_table)
        for part_index, chunk in enumerate(chunks, start=1):
            chunk_records = [row.record for row in chunk]
            headers, second, colspec, show_c, show_d = _base_headers_and_colspec(chunk_records, options)
            headers.extend(['allowed $\\lambda,L$', f'${_profile_symbol(chunk_records)}$'])
            second.extend([format_allowed_lambda_header(chunk_records, options), ''])
            colspec.extend(['p{2.2cm}', 'c'])
            _, codes, count_rule = _append_count_header(headers, second, colspec, chunk_records, options, 'ranges')
            if options.show_tags:
                headers.append('tag')
                second.append('')
                colspec.append('p{1.8cm}')
            ncols = len(colspec)
            table_caption = caption
            if len(context_groups) > 1:
                table_caption += f' Context group {group_index} of {len(context_groups)}.'
            table_caption = _caption_for_part(table_caption, part_index, len(chunks))
            table_label = _sanitize_label(label)
            if len(context_groups) > 1:
                table_label += f'_context_{group_index}'
            table_label = _label_for_part(table_label, part_index, len(chunks))
            _emit_table_preamble(parts, caption=table_caption, label=table_label, colspec=''.join(colspec), size='scriptsize', tabcolsep='2pt', arraystretch='1.12', resizebox=resizebox)
            context = _context_header(chunk_records, options, ncols)
            _emit_two_level_header(parts, headers, second, [count_rule] if count_rule else [], context)
            for range_row in chunk:
                record = range_row.record
                color = _row_color_command(record, options)
                if color:
                    parts.append(color)
                cells = _base_cells(record, options, show_c, show_d)
                cells.extend([f'${format_allowed_lambda_L(record, options)}$', f'${format_value(range_row.displayed_value)}$'])
                _append_count_cells(cells, record, codes)
                if options.show_tags:
                    cells.append(f'${format_tags(record.tags)}$' if record.tags else '')
                parts.append(' & '.join(cells) + ' \\\\')
            _emit_table_end(parts, resizebox=resizebox)
    Path(output).write_text('\n'.join(parts), encoding='utf-8')

def _channel_group_key(record):
    return (record.N, tuple((block.key() for block in record.nu_blocks)))

def write_wide_table(records, output, *, caption, label, max_rows_per_table=24, options=FormatOptions(), max_L_columns=25, allow_huge_wide=False, resizebox=True):
    if not records:
        raise TableInputError('No records to write')
    for record in records:
        if not record.profile.known():
            raise TableInputError('Wide layout requires exact per-L values for every row')
    channel_groups = OrderedDict()
    for record in sorted(records, key=_record_sort_key):
        channel_groups.setdefault(_channel_group_key(record), []).append(record)
    parts = _common_header_comment(records, options)
    label = _sanitize_label(label)
    group_count = len(channel_groups)
    for group_index, (_group_key, group_records) in enumerate(channel_groups.items(), start=1):
        point_maps = [record.profile.to_points(limit=max_L_columns * 20 if not allow_huge_wide else 1000000) for record in group_records]
        observed = sorted(set().union(*(points.keys() for points in point_maps)))
        L_values = [0] if not observed else list(range(0, max(observed) + 1))
        if len(L_values) > max_L_columns and (not allow_huge_wide):
            raise TableInputError(f'Wide layout would require {len(L_values)} explicit L columns. Use --layout ranges/compact or raise --max-wide-columns with --allow-huge-wide.')
        chunks = _table_parts(group_records, max_rows_per_table)
        for chunk_index, chunk in enumerate(chunks, start=1):
            headers, second, colspec_list, show_c, show_d = _base_headers_and_colspec(chunk, options)
            prefix_count = len(colspec_list)
            profile_start = prefix_count + 1
            headers.append(f'\\multicolumn{{{len(L_values)}}}{{c}}{{${_profile_symbol(chunk)}$}}')
            colspec_list.extend(['c'] * len(L_values))
            profile_rule = (profile_start, profile_start + len(L_values) - 1)
            second.extend([f'$L={L}$' for L in L_values])
            _, codes, count_rule = _append_count_header(headers, second, colspec_list, chunk, options, 'wide')
            if options.show_tags:
                headers.append('tag')
                second.append('')
                colspec_list.append('p{1.8cm}')
            ncols = len(colspec_list)
            table_caption = caption
            if group_count > 1:
                table_caption += f' Channel group {group_index} of {group_count}.'
            table_caption = _caption_for_part(table_caption, chunk_index, len(chunks))
            table_label = label + (f'_channel_{group_index}' if group_count > 1 else '')
            table_label = _label_for_part(table_label, chunk_index, len(chunks))
            _emit_table_preamble(parts, caption=table_caption, label=table_label, colspec=''.join(colspec_list), size='tiny', tabcolsep='1.5pt', arraystretch='1.08', resizebox=resizebox)
            context = _context_header(chunk, options, ncols)
            rules = [profile_rule] + ([count_rule] if count_rule else [])
            _emit_two_level_header(parts, headers, second, rules, context)
            for record in chunk:
                color = _row_color_command(record, options)
                if color:
                    parts.append(color)
                points = record.profile.to_points(limit=max(100000, len(L_values) * 2))
                cells = _base_cells(record, options, show_c, show_d)
                cells.extend((str(points.get(L, 0)) for L in L_values))
                _append_count_cells(cells, record, codes)
                if options.show_tags:
                    cells.append(f'${format_tags(record.tags)}$' if record.tags else '')
                parts.append(' & '.join(cells) + ' \\\\')
            _emit_table_end(parts, resizebox=resizebox)
    Path(output).write_text('\n'.join(parts), encoding='utf-8')

def write_breakdown_table(records, out_tex, caption, label, max_rows_per_table=32, *, options=FormatOptions()):
    write_compact_table(records, out_tex, caption=caption, label=label, max_rows_per_table=max_rows_per_table, options=options)

def write_breakdown_table_wide(records, out_tex, caption, label, max_rows_per_table=24, *, options=FormatOptions(), max_L_columns=25, allow_huge_wide=False):
    write_wide_table(records, out_tex, caption=caption, label=label, max_rows_per_table=max_rows_per_table, options=options, max_L_columns=max_L_columns, allow_huge_wide=allow_huge_wide)

def choose_layout(records, *, max_wide_columns=25, wide_max_rank=8):
    if not records:
        return 'summary'
    if any((not record.profile.known() for record in records)):
        return 'summary'
    max_rank = max((record.N for record in records))
    max_L = max((record.profile.max_L() or 0 for record in records))
    if max_rank <= wide_max_rank and max_L + 1 <= max_wide_columns:
        return 'wide'
    return 'ranges'

def default_caption(records, layout):
    ranks = sorted({record.N for record in records})
    rank_text = f'rank ${ranks[0]}$' if len(ranks) == 1 else 'the listed ranks'
    if layout == 'summary':
        return f'Representation decomposition and code-reported counts grouped by $(\\boldsymbol\\kappa,\\boldsymbol\\Lambda,\\lambda,L)$ for {rank_text}.'
    if layout == 'wide':
        return f'Young--rotation decomposition resolved by $(\\boldsymbol\\kappa,\\boldsymbol\\Lambda,\\lambda,L)$ for {rank_text}.'
    if layout == 'ranges':
        return f'Young--rotation decomposition on compact angular-momentum ranges, resolved by $(\\boldsymbol\\kappa,\\boldsymbol\\Lambda,\\lambda,L)$ for {rank_text}.'
    return f'Compact Young--rotation profiles resolved by $(\\boldsymbol\\kappa,\\boldsymbol\\Lambda,\\lambda,L)$ for {rank_text}.'

def default_label(records, layout):
    ranks = sorted({record.N for record in records})
    rank_part = str(ranks[0]) if len(ranks) == 1 else 'multiple'
    return f'tab:ye3t_external_rank_{rank_part}_{layout}'

def write_standalone_wrapper(snippet_path, output_path):
    snippet = Path(snippet_path).read_text(encoding='utf-8')
    document = '\\documentclass[reprint,aps,prb,floatfix]{revtex4-2}\n\\usepackage{amsmath}\n\\usepackage{booktabs}\n\\usepackage{graphicx}\n\\usepackage[table]{xcolor}\n\\usepackage{ytableau}\n\\ytableausetup{boxsize=0.45em}\n\\begin{document}\n' + snippet + '\n' + '\\end{document}' + '\n'
    Path(output_path).write_text(document, encoding='utf-8')

def _parse_count_codes_arg(value):
    return tuple((piece.strip() for piece in value.split(',') if piece.strip()))


TABLE_FORMATTER_CONFIG_DEFAULTS = {
    "input_format": "auto",
    "defaults": {
        "N": None,
        "eta": None,
        "l": None,
        "nu": None,
        "kappa": None,
        "Lambda": None,
        "lambda": None,
        "profile_kind": None,
        "count_scope": None,
        "sector": None,
        "tags": None,
    },
    "layout": "auto",
    "caption": None,
    "label": None,
    "max_rows": 24,
    "max_wide_columns": 25,
    "wide_max_rank": 8,
    "allow_huge_wide": False,
    "partition_style": "auto",
    "kappa_style": None,
    "lambda_style": None,
    "diagram_max_rank": 8,
    "ydiagram_max_cells": 256,
    "ydiagram_max_rows": 256,
    "max_channel_blocks": 8,
    "columns": "full",
    "rank_placement": "column",
    "group_placement": "column",
    "context_header": False,
    "omit_N": False,
    "omit_group": False,
    "show_input_channels": False,
    "omit_nu": False,
    "omit_kappa": False,
    "omit_block_lambda": False,
    "omit_parent_lambda": False,
    "show_d_blocks": False,
    "show_cd": "auto",
    "count_codes": "YE3T",
    "count_scope": "auto",
    "count_symbol": None,
    "value_label": None,
    "profile_columns": "omit",
    "split_by_rank": False,
    "group_resolved_rows": False,
    "show_tags": False,
    "omit_tags": False,
    "row_colors": "auto",
    "no_row_colors": False,
    "ace_row_color": "blue!8",
    "sign_row_color": "green!10",
    "mixed_row_color": "",
    "no_resizebox": False,
    "infer_trivial_labels": False,
    "no_auto_tags": False,
    "no_validate": False,
    "duplicate_policy": "sum",
    "normalized_out": None,
    "standalone_out": None,
    "print_layout": False,
}


TABLE_FORMATTER_ALLOWED_OPTIONS = OrderedDict([
    ("input_format", ("auto", "csv", "json", "jsonl", "python", "text", "kv", "printed")),
    ("layout", ("auto", "wide", "ranges", "compact", "summary")),
    ("partition_style", ("auto", "partition", "compact", "ydiagram", "tuple")),
    ("kappa_style", (None, "auto", "partition", "compact", "ydiagram", "tuple")),
    ("lambda_style", (None, "auto", "partition", "compact", "ydiagram", "tuple")),
    ("columns", ("full", "ace-comparison", "minimal")),
    ("rank_placement", ("column", "header", "none")),
    ("group_placement", ("column", "header", "none")),
    ("show_cd", ("auto", "yes", "no")),
    ("count_scope", ("auto", "lambda", "global", "row")),
    ("profile_columns", ("omit", "compact", "subcolumns")),
    ("row_colors", ("auto", "off")),
    ("duplicate_policy", ("sum", "error", "last")),
])


def _deepcopy_config(value):
    if isinstance(value, dict):
        return {key: _deepcopy_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deepcopy_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deepcopy_config(item) for item in value)
    return value


def table_formatter_config_defaults():
    return _deepcopy_config(TABLE_FORMATTER_CONFIG_DEFAULTS)


def table_formatter_allowed_options():
    return OrderedDict((key, tuple(value)) for key, value in TABLE_FORMATTER_ALLOWED_OPTIONS.items())


def _merge_config(base, override):
    merged = _deepcopy_config(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = _deepcopy_config(value)
    return merged


def _validate_choice(config, key):
    if key not in TABLE_FORMATTER_ALLOWED_OPTIONS:
        return
    value = config.get(key)
    allowed = TABLE_FORMATTER_ALLOWED_OPTIONS[key]
    if value not in allowed:
        raise TableInputError(f"table formatter option {key}={value!r} must be one of {allowed!r}")


def normalize_table_formatter_config(config=None):
    normalized = _merge_config(TABLE_FORMATTER_CONFIG_DEFAULTS, config or {})
    for key in TABLE_FORMATTER_ALLOWED_OPTIONS:
        _validate_choice(normalized, key)
    return normalized


def _records_from_config(input_path, config):
    defaults = {key: value for key, value in dict(config.get("defaults") or {}).items() if value is not None}
    return load_records(
        input_path,
        input_format=config["input_format"],
        infer_trivial_labels=bool(config["infer_trivial_labels"]),
        validate=not bool(config["no_validate"]),
        auto_tags=not bool(config["no_auto_tags"]),
        duplicate_policy=config["duplicate_policy"],
        defaults=defaults,
    )


def _format_options_from_config(config):
    rank_placement = config["rank_placement"]
    group_placement = config["group_placement"]
    if config["context_header"]:
        rank_placement = group_placement = "header"
    if config["omit_N"]:
        rank_placement = "none"
    if config["omit_group"]:
        group_placement = "none"
    show_nu = not config["omit_nu"]
    show_kappa = not config["omit_kappa"]
    show_block_lambda = not config["omit_block_lambda"]
    show_parent_lambda = not config["omit_parent_lambda"]
    show_cd = config["show_cd"]
    if config["columns"] == "ace-comparison":
        show_kappa = False
        show_block_lambda = False
        show_parent_lambda = False
        show_cd = "no"
    elif config["columns"] == "minimal":
        show_kappa = False
        show_block_lambda = False
        show_parent_lambda = False
        show_cd = "no"
        if group_placement == "column":
            group_placement = "none"
    row_colors = "off" if config["no_row_colors"] else config["row_colors"]
    count_codes = config["count_codes"]
    if isinstance(count_codes, (list, tuple)):
        count_codes = ",".join(str(item) for item in count_codes)
    return FormatOptions(
        partition_style=config["partition_style"],
        kappa_style=config["kappa_style"],
        lambda_style=config["lambda_style"],
        diagram_max_rank=int(config["diagram_max_rank"]),
        ydiagram_max_cells=int(config["ydiagram_max_cells"]),
        ydiagram_max_rows=int(config["ydiagram_max_rows"]),
        max_channel_blocks=int(config["max_channel_blocks"]),
        rank_placement=rank_placement,
        group_placement=group_placement,
        show_nu=show_nu,
        show_kappa=show_kappa,
        show_block_lambda=show_block_lambda,
        show_parent_lambda=show_parent_lambda,
        show_tags=bool(config["show_tags"]) and not bool(config["omit_tags"]),
        row_colors=row_colors,
        ace_row_color=config["ace_row_color"],
        sign_row_color=config["sign_row_color"],
        mixed_row_color=config["mixed_row_color"],
        show_cd=show_cd,
        show_d_blocks=bool(config["show_d_blocks"]),
        show_input_channels=bool(config["show_input_channels"]),
        count_codes=_parse_count_codes_arg(count_codes),
        count_scope=config["count_scope"],
        count_symbol=config["count_symbol"] or config["value_label"],
        profile_columns=config["profile_columns"],
        split_by_rank=bool(config["split_by_rank"]),
        group_resolved_rows=bool(config["group_resolved_rows"]),
    )


def write_latex_table_from_config(input_path, output_path, config=None):
    config = normalize_table_formatter_config(config)
    records = _records_from_config(input_path, config)
    if not records:
        raise TableInputError("Input contained no data rows")
    layout = config["layout"]
    if layout == "auto":
        layout = choose_layout(records, max_wide_columns=int(config["max_wide_columns"]), wide_max_rank=int(config["wide_max_rank"]))
    if config["print_layout"]:
        print(f"chosen layout: {layout}", file=sys.stderr)
    options = _format_options_from_config(config)
    caption = config["caption"] or default_caption(records, layout)
    label = config["label"] or default_label(records, layout)
    common = dict(records=records, output=output_path, caption=caption, label=label, max_rows_per_table=int(config["max_rows"]), options=options)
    if layout == "summary":
        write_summary_table(**common, resizebox=not bool(config["no_resizebox"]))
    elif layout == "compact":
        write_compact_table(**common, resizebox=not bool(config["no_resizebox"]))
    elif layout == "ranges":
        write_range_table(**common, resizebox=not bool(config["no_resizebox"]))
    elif layout == "wide":
        write_wide_table(**common, max_L_columns=int(config["max_wide_columns"]), allow_huge_wide=bool(config["allow_huge_wide"]), resizebox=not bool(config["no_resizebox"]))
    else:
        raise TableInputError(f"Unsupported layout {layout!r}")
    if config["normalized_out"]:
        write_normalized_jsonl(records, config["normalized_out"])
    if config["standalone_out"]:
        write_standalone_wrapper(output_path, config["standalone_out"])
    return output_path


def _column_settings(settings, table_settings):
    base = dict(settings.get("column_settings", {}))
    base.update(dict(table_settings.get("column_settings", {})))
    if "columns" in table_settings:
        base["preset"] = table_settings["columns"]
    return base


def _config_for_table(settings, table_settings):
    config = dict(settings.get("formatter_options", {}))
    for key in TABLE_FORMATTER_CONFIG_DEFAULTS:
        if key in settings and key != "defaults":
            config[key] = settings[key]
    for key, value in dict(table_settings or {}).items():
        if key in TABLE_FORMATTER_CONFIG_DEFAULTS and key != "defaults":
            config[key] = value
    column_settings = _column_settings(settings, table_settings or {})
    if "preset" in column_settings:
        config["columns"] = column_settings["preset"]
    for key in ("rank_placement", "group_placement", "show_input_channels", "show_d_blocks", "show_tags"):
        if key in column_settings:
            config[key] = column_settings[key]
    defaults = dict(settings.get("formatter_defaults", {}))
    defaults.update(dict((table_settings or {}).get("formatter_defaults", {})))
    if defaults:
        config["defaults"] = defaults
    return config


def write_latex_tables(v6_json, resolved_v6_json, output_dir, settings):
    table_specs = settings.get("latex_tables") or [{"filename": settings["latex_table"]}]
    written = []
    skipped = []
    for table_spec in table_specs:
        tex_path = Path(output_dir) / table_spec.get("filename", settings["latex_table"])
        source = table_spec.get("row_source", "aggregate")
        if source == "resolved" and not settings.get("_resolved_rows_available", True):
            skipped.append((tex_path, "no resolved kappa/Lambda rows were produced"))
            continue
        source_json = resolved_v6_json if source == "resolved" else v6_json
        write_latex_table_from_config(source_json, tex_path, _config_for_table(settings, table_spec))
        written.append(tex_path)
    return written, skipped
