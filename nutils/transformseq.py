# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""The transformseq module."""

from . import types, numeric, util, transform, element, warnings, evaluable
from .elementseq import References
from typing import Tuple, Optional
import abc, itertools, operator, numpy, functools

class Transforms(types.Singleton):
  '''Abstract base class for a sequence of :class:`~nutils.transform.TransformItem` tuples.

  This class resembles to some extent a plain :class:`tuple`: the
  class supports indexing, iterating and has an :meth:`index` method.  In
  addition the class supports the :meth:`index_with_tail` method which can be
  used to find the index of a transform given the transform plus any number of
  child transformations.

  The transforms in this sequence must satisfy the following condition: any
  transform must not start with any other transform in the same sequence.

  Parameters
  ----------
  todims : :class:`int`
      The dimensions all transform chains in this sequence map to.
  fromdim : :class:`int`
      The dimension all transforms in this sequence map from.

  Attributes
  ----------
  todim : :class:`int`
      The dimension all transform chains in this sequence map to.
  todims : :class:`tuple` of :class:`int`
      The dimensions of the separate transform chains.
  fromdim : :class:`int`
      The dimension all transforms in this sequence map from.

  Notes
  -----
  Subclasses must implement :meth:`__getitem__`, :meth:`__len__` and
  :meth:`index_with_tail`.
  '''

  __slots__ = 'todims', 'todim', 'fromdim'

  @types.apply_annotations
  def __init__(self, todims:types.tuple[types.strictint], fromdim:types.strictint):
    self.todims = todims
    self.todim = sum(todims)
    self.fromdim = fromdim
    super().__init__()

  @property
  def fromdims(self):
    warnings.deprecation('`fromdims` has been renamed to `fromdim`')
    return self.fromdim

  @abc.abstractmethod
  def __len__(self):
    '''Return ``len(self)``.'''

    raise NotImplementedError

  def __getitem__(self, index):
    '''Return ``self[index]``.'''

    if numeric.isint(index):
      raise NotImplementedError
    elif isinstance(index, slice):
      index = range(len(self))[index]
      if index == range(len(self)):
        return self
      if index.step < 0:
        raise NotImplementedError('reordering the sequence is not yet implemented')
      return MaskedTransforms(self, numpy.arange(index.start, index.stop, index.step))
    elif numeric.isintarray(index):
      if index.ndim != 1:
        raise IndexError('invalid index')
      if numpy.any(numpy.less(index, 0)) or numpy.any(numpy.greater_equal(index, len(self))):
        raise IndexError('index out of range')
      dindex = numpy.diff(index)
      if len(index) == len(self) and (len(self) == 0 or (index[0] == 0 and numpy.all(numpy.equal(dindex, 1)))):
        return self
      if numpy.any(numpy.equal(dindex, 0)):
        raise ValueError('repeating an element is not allowed')
      if not numpy.all(numpy.greater(dindex, 0)):
        s = numpy.argsort(index)
        return ReorderedTransforms(self[index[s]], numpy.argsort(s))
      if len(index) == 0:
        return EmptyTransforms(self.todims, self.fromdim)
      if len(index) == len(self):
        return self
      return MaskedTransforms(self, index)
    elif numeric.isboolarray(index):
      if index.shape != (len(self),):
        raise IndexError('mask has invalid shape')
      if not numpy.any(index):
        return EmptyTransforms(self.todims, self.fromdim)
      if numpy.all(index):
        return self
      index, = numpy.where(index)
      return MaskedTransforms(self, index)
    else:
      raise IndexError('invalid index')

  @abc.abstractmethod
  def index_with_tail(self, trans):
    '''Return the index of ``trans[:n]`` and the tail ``trans[n:]``.

    Find the index of a transform in this sequence given the transform plus any
    number of child transforms.  In other words: find ``index`` such that
    ``self[index] == trans[:n]`` for some ``n``.  Note that there is either
    exactly one ``index`` satisfying this condition, or none, due to the
    restrictions of the transforms in a :class:`Transforms` object.

    Parameters
    ----------
    trans : :class:`tuple` of :class:`nutils.transform.TransformItem` objects
        The transform to find up to a possibly empty tail.

    Returns
    -------
    index : :class:`int`
        The index of ``trans`` without tail in this sequence.
    tail : :class:`tuple` of :class:`nutils.transform.TransformItem` objects
        The tail: ``trans[len(self[index]):]``.

    Raises
    ------
    :class:`ValueError`
        if ``trans`` is not found.

    Example
    -------

    Consider the following plain sequence of two shift transforms:

    >>> from nutils.transform import Shift, Scale, TransformChain, TransformChains
    >>> transforms = PlainTransforms([TransformChain(Shift([0.])), TransformChain(Shift([1.]))], 1, 1)

    Calling :meth:`index_with_tail` with the first transform gives index ``0``
    and no tail:

    >>> transforms.index_with_tail(TransformChains(TransformChain(Shift([0.]))))
    (0, TransformChains(TransformChain(todim=1)))

    Calling with an additional scale gives:

    >>> transforms.index_with_tail(TransformChains(TransformChain(Shift([0.]), Scale(0.5, [0.]))))
    (0, TransformChains(TransformChain(Scale([0]+0.5*x))))
    '''

    raise NotImplementedError

  def __iter__(self):
    '''Implement ``iter(self)``.'''

    for i in range(len(self)):
      yield self[i]

  def index(self, trans):
    '''Return the index of ``trans``.

    Parameters
    ----------
    trans : :class:`~nutils.transform.TransformChain`

    Returns
    -------
    index : :class:`int`
        The index of ``trans`` in this sequence.

    Raises
    ------
    :class:`ValueError`
        if ``trans`` is not found.

    Example
    -------

    Consider the following plain sequence of two shift transforms:

    >>> from nutils.transform import Shift, Scale, TransformChain, TransformChains
    >>> transforms = PlainTransforms([TransformChain(Shift([0.])), TransformChain(Shift([1.]))], 1, 1)

    Calling :meth:`index` with the first transform gives index ``0``:

    >>> transforms.index(TransformChains(TransformChain(Shift([0.]))))
    0

    Calling with an additional scale raises an exception, because the transform
    is not present in ``transforms``.

    >>> transforms.index(TransformChains(TransformChain(Shift([0.]), Scale(0.5, [0.]))))
    Traceback (most recent call last):
      ...
    ValueError: TransformChains(TransformChain(Shift([0]+x), Scale([0]+0.5*x))) not in sequence of transforms
    '''

    index, tail = self.index_with_tail(trans)
    if any(tail):
      raise ValueError('{!r} not in sequence of transforms'.format(trans))
    return index

  def contains(self, trans):
    '''Return ``trans`` in ``self``.

    Parameters
    ----------
    trans : :class:`~nutils.transform.TransformChain`

    Returns
    -------
    :class:`bool`
        ``True`` if ``trans`` is contained in this sequence of transforms, i.e.
        if :meth:`index` returns without :class:`ValueError`, otherwise
        ``False``.
    '''

    try:
      self.index(trans)
    except ValueError:
      return False
    else:
      return True

  __contains__ = contains

  def contains_with_tail(self, trans):
    '''Return ``trans[:n]`` in ``self`` for some ``n``.

    Parameters
    ----------
    trans : :class:`~nutils.transform.TransformChain`

    Returns
    -------
    :class:`bool`
        ``True`` if a head of ``trans`` is contained in this sequence
        of transforms, i.e. if :meth:`index_with_tail` returns without
        :class:`ValueError`, otherwise ``False``.
    '''

    try:
      self.index_with_tail(trans)
    except ValueError:
      return False
    else:
      return True

  def refined(self, references):
    '''Return the sequence of refined transforms given ``references``.

    Parameters
    ----------
    references : :class:`~nutils.elementseq.References`
        A sequence of references matching this sequence of transforms.

    Returns
    -------
    :class:`Transforms`
        The sequence of refined transforms::

            (trans.append(ctrans) for trans, ref in zip(self, references) for ctrans in ref.child_transforms)
    '''

    if references.isuniform:
      return UniformDerivedTransforms(self, references[0], 'child_transforms', self.fromdim)
    else:
      return DerivedTransforms(self, references, 'child_transforms', self.fromdim)

  def edges(self, references):
    '''Return the sequence of edge transforms given ``references``.

    Parameters
    ----------
    references : :class:`~nutils.elementseq.References`
        A sequence of references matching this sequence of transforms.

    Returns
    -------
    :class:`Transforms`
        The sequence of edge transforms::

            (trans.append(etrans) for trans, ref in zip(self, references) for etrans in ref.edge_transforms)
    '''

    if references.isuniform:
      return UniformDerivedTransforms(self, references[0], 'edge_transforms', self.fromdim-1)
    else:
      return DerivedTransforms(self, references, 'edge_transforms', self.fromdim-1)

  def __add__(self, other):
    '''Return ``self+other``.'''

    if not isinstance(other, Transforms) or self.fromdim != other.fromdim:
      return NotImplemented
    return chain((self, other), self.todims, self.fromdim)

  def unchain(self):
    '''Iterator of unchained :class:`Transforms` items.

    Yields
    ------
    :class:`Transforms`
        Unchained items.
    '''

    yield self

  def get_evaluable(self, index: evaluable.Array) -> transform.EvaluableTransformChains:
    '''Return the evaluable transform chain at the given index.

    Parameter
    ---------
    index : a scalar, integer :class:`nutils.evaluable.Array`
        The index of the transform chains to return.

    Returns
    -------
    :class:`nutils.transform.EvaluableTransformChain`
        The evaluable transform chains at the given ``index``.
    '''

    return _EvaluableTransformChainsFromSequence(self, index)

  def evaluable_index_with_tail(self, chains: transform.EvaluableTransformChains) -> Tuple[evaluable.Array, transform.EvaluableTransformChains]:
    index_tails = _EvaluableIndexWithTails(self, chains)
    index = evaluable.ArrayFromTuple(index_tails, 0, (), int)
    tails = _EvaluableTransformChainsFromTuple(index_tails, 1, self.get_evaluable(index).fromdims, chains.fromdims)
    return index, tails

stricttransforms = types.strict[Transforms]

class EmptyTransforms(Transforms):
  '''An empty sequence.'''

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    raise IndexError('index out of range')

  def __len__(self):
    return 0

  def index_with_tail(self, trans):
    raise ValueError

  def index(self, trans):
    raise ValueError

  def contains_with_tail(self, trans):
    return False

  def contains(self, trans):
    return False

  __contains__ = contains

class PlainTransforms(Transforms):
  '''A general purpose implementation of :class:`Transforms`.

  Use this class only if there exists no specific implementation of
  :class:`Transforms` for the transforms at hand.

  Parameters
  ----------
  transforms : :class:`tuple` of :class:`~nutils.transform.TransformChain`
      The sequence of transforms.
  fromdim : :class:`int`
      The dimension all ``transforms`` map from.
  '''

  __slots__ = '_transforms', '_sorted', '_indices'

  @types.apply_annotations
  def __init__(self, transforms:types.tuple[transform.canonical], todim:types.strictint, fromdim:types.strictint):
    assert all(isinstance(chain, transform.TransformChain) for chain in transforms)
    transforms_todim = set(trans.todim for trans in transforms)
    transforms_fromdim = set(trans.fromdim for trans in transforms)
    if not (transforms_todim <= {todim}):
      raise ValueError('expected transforms with todim={}, but got {}'.format(todim, transforms_todim))
    if not (transforms_fromdim <= {fromdim}):
      raise ValueError('expected transforms with fromdim={}, but got {}'.format(fromdim, transforms_fromdim))
    self._transforms = transforms
    self._sorted = numpy.empty([len(self._transforms)], dtype=object)
    for i, trans in enumerate(self._transforms):
      self._sorted[i] = tuple(map(id, trans))
    self._indices = numpy.argsort(self._sorted)
    self._sorted = self._sorted[self._indices]
    super().__init__((todim,), fromdim)

  def __iter__(self):
    return map(transform.TransformChains, self._transforms)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    return transform.TransformChains(self._transforms[numeric.normdim(len(self), index)])

  def __len__(self):
    return len(self._transforms)

  def index_with_tail(self, chains):
    if len(chains) != 1:
      raise ValueError('Expected a product of one chain but got {}.'.format(len(chains)))
    trans, = chains
    trans, orig_trans = trans.promote(self.fromdim), trans
    transid_array = numpy.empty((), dtype=object)
    transid_array[()] = transid = tuple(map(id, trans))
    i = numpy.searchsorted(self._sorted, transid_array, side='right') - 1
    if i < 0:
      raise ValueError('{!r} not in sequence of transforms'.format(orig_trans))
    match = self._sorted[i]
    if transid[:len(match)] != match:
      raise ValueError('{!r} not in sequence of transforms'.format(orig_trans))
    return self._indices[i], transform.TransformChains(trans[len(match):])

class IdentifierTransforms(Transforms):
  '''A sequence of :class:`nutils.transform.Identifier` singletons.

  Every identifier is instantiated with three arguments: the dimension, the
  name string, and an integer index matching its position in the sequence.

  Parameters
  ----------
  ndims : :class:`int`
      Dimension of the transformation.
  name : :class:`str`
      Identifying name string.
  length : :class:`int`
      Length of the sequence.
  '''

  __slots__ = '_name', '_length'

  @types.apply_annotations
  def __init__(self, ndims:types.strictint, name:str, length:int):
    self._name = name
    self._length = length
    super().__init__((ndims,), ndims)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    index = int(index) # make sure that index is a Python integer rather than numpy.intxx
    return transform.TransformChains(transform.TransformChain(transform.Identifier(self.fromdim, (self._name, numeric.normdim(self._length, index)))))

  def __len__(self):
    return self._length

  def index_with_tail(self, chains):
    if len(chains) != 1:
      raise ValueError('Expected a product of one chain but got {}.'.format(len(chains)))
    trans, = chains
    root = trans[0]
    if root.fromdim == self.fromdim and isinstance(root, transform.Identifier) and isinstance(root.token, tuple) and len(root.token) == 2 and root.token[0] == self._name and 0 <= root.token[1] < self._length:
      return root.token[1], transform.TransformChains(trans[1:])
    raise ValueError

class Axis(types.Singleton):
  '''Base class for axes of :class:`~nutils.topology.StructuredTopology`.'''

  __slots__ = 'i', 'j', 'mod'

  def __init__(self, i:types.strictint, j:types.strictint, mod:types.strictint):
    assert i <= j
    self.i = i
    self.j = j
    self.mod = mod

  def __len__(self):
    return self.j - self.i

  def unmap(self, index):
    ielem = index - self.i
    if self.mod:
      ielem %= self.mod
    if not 0 <= ielem < len(self):
      raise ValueError
    return ielem

  def map(self, ielem):
    assert 0 <= ielem < len(self)
    index = self.i + ielem
    if self.mod:
      index %= self.mod
    return index

class DimAxis(Axis):

  __slots__ = 'isperiodic'
  isdim = True

  @types.apply_annotations
  def __init__(self, i:types.strictint, j:types.strictint, mod:types.strictint, isperiodic:bool):
    super().__init__(i, j, mod)
    self.isperiodic = isperiodic

  @property
  def refined(self):
    return DimAxis(self.i*2, self.j*2, self.mod*2, self.isperiodic)

  def opposite(self, ibound):
    return self

  def getitem(self, s):
    if not isinstance(s, slice):
      raise NotImplementedError
    if s == slice(None):
      return self
    start, stop, stride = s.indices(self.j - self.i)
    assert stride == 1
    assert stop > start
    return DimAxis(self.i+start, self.i+stop, mod=self.mod, isperiodic=False)

  def boundaries(self, ibound):
    if not self.isperiodic:
      yield IntAxis(self.i, self.i+1, self.mod, ibound, side=False)
      yield IntAxis(self.j-1, self.j, self.mod, ibound, side=True)

  def intaxis(self, ibound, side):
    return IntAxis(self.i-side+1-self.isperiodic, self.j-side, self.mod, ibound, side)

class IntAxis(Axis):

  __slots__ = 'ibound', 'side'
  isdim = False

  @types.apply_annotations
  def __init__(self, i:types.strictint, j:types.strictint, mod:types.strictint, ibound:types.strictint, side:bool):
    super().__init__(i, j, mod)
    self.ibound = ibound
    self.side = side

  @property
  def refined(self):
    return IntAxis(self.i*2+self.side, self.j*2+self.side-1, self.mod*2, self.ibound, self.side)

  def opposite(self, ibound):
    return IntAxis(self.i+2*self.side-1, self.j+2*self.side-1, self.mod, self.ibound, not self.side) if ibound == self.ibound else self

  def boundaries(self, ibound):
    return ()

class StructuredTransforms(Transforms):
  '''Transforms sequence for :class:`~nutils.topology.StructuredTopology`.

  Parameters
  ----------
  root : :class:`~nutils.transform.TransformItem`
      Root transform of the :class:`~nutils.topology.StructuredTopology`.
  axes : :class:`tuple` of :class:`Axis` objects
      The axes defining the :class:`~nutils.topology.StructuredTopology`.
  nrefine : :class:`int`
      Number of structured refinements.
  '''

  __slots__ = '_root', '_axes', '_nrefine', '_etransforms', '_ctransforms', '_cindices'

  @types.apply_annotations
  def __init__(self, root:transform.stricttransformitem, axes:types.tuple[types.strict[Axis]], nrefine:types.strictint):
    self._root = root
    self._axes = axes
    self._nrefine = nrefine

    ref = element.LineReference()**len(self._axes)
    self._ctransforms = numeric.asobjvector(ref.child_transforms).reshape((2,)*len(self._axes))
    self._cindices = {t: numpy.array(i, dtype=int) for i, t in numpy.ndenumerate(self._ctransforms)}

    etransforms = []
    rmdims = numpy.zeros(len(axes), dtype=bool)
    for order, side, idim in sorted((axis.ibound, axis.side, idim) for idim, axis in enumerate(axes) if not axis.isdim):
      ref = util.product(element.getsimplex(0 if rmdim else 1) for rmdim in rmdims)
      iedge = (idim - rmdims[:idim].sum()) * 2 + 1 - side
      etransforms.append(ref.edge_transforms[iedge])
      rmdims[idim] = True
    self._etransforms = tuple(etransforms)

    super().__init__((root.todim,), sum(axis.isdim for axis in self._axes))

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    index = numeric.normdim(len(self), index)
    # Decompose index into indices per dimension on the nrefined level.
    indices = []
    for axis in reversed(self._axes):
      index, rem = divmod(index, len(axis))
      indices.insert(0, axis.map(rem))
    assert index == 0
    # Create transform.
    ctransforms = []
    indices = numpy.asarray(indices, dtype=int)
    for i in range(self._nrefine):
      indices, r = divmod(indices, self._ctransforms.shape)
      ctransforms.insert(0, self._ctransforms[tuple(r)])
    trans0 = transform.Shift(types.frozenarray(indices, dtype=float, copy=False))
    return transform.TransformChains(transform.TransformChain(self._root, trans0, *ctransforms, *self._etransforms))

  def __len__(self):
    return util.product(map(len, self._axes))

  def index_with_tail(self, chains):
    if len(chains) != 1:
      raise ValueError('Expected a product of one chain but got {}.'.format(len(chains)))
    trans, = chains
    if len(trans) < 2 + self._nrefine + len(self._etransforms):
      raise ValueError

    root, shift, tail = trans[0], trans[1], trans[2:].uppermost
    if root != self._root:
      raise ValueError

    if not isinstance(shift, transform.Shift) or len(shift.offset) != len(self._axes) or not numpy.equal(shift.offset.astype(int), shift.offset).all():
      raise ValueError
    indices = numpy.array(shift.offset, dtype=int)

    # Match child transforms.
    for item in tail[:self._nrefine]:
      try:
        indices = indices*2 + self._cindices[item]
      except KeyError:
        raise ValueError

    # Check index boundaries and flatten.
    flatindex = 0
    for index, axis in zip(indices, self._axes):
      flatindex = flatindex*len(axis) + axis.unmap(index)

    # Promote the remainder and match the edge transforms.
    tail = tail[self._nrefine:].promote(self.fromdim)
    if tuple(tail[:len(self._etransforms)]) != self._etransforms:
      raise ValueError
    tail = tail[len(self._etransforms):]

    return flatindex, transform.TransformChains(tail)

class MaskedTransforms(Transforms):
  '''An order preserving subset of another :class:`Transforms` object.

  Parameters
  ----------
  parent : :class:`Transforms`
      The transforms to subset.
  indices : one-dimensional array of :class:`int`\\s
      The strict monotonic increasing indices of ``parent`` transforms to keep.
  '''

  __slots__ = '_parent', '_mask', '_indices'

  @types.apply_annotations
  def __init__(self, parent:stricttransforms, indices:types.arraydata):
    assert indices.dtype == int
    self._parent = parent
    self._indices = numpy.asarray(indices)
    super().__init__(parent.todims, parent.fromdim)

  def __iter__(self):
    for itrans in self._indices:
      yield self._parent[int(itrans)]

  def __getitem__(self, index):
    if numeric.isintarray(index) and index.ndim == 1 and numpy.any(numpy.less(index, 0)):
      raise IndexError('index out of bounds')
    return self._parent[self._indices[index]]

  def __len__(self):
    return len(self._indices)

  def index_with_tail(self, trans):
    parent_index, tail = self._parent.index_with_tail(trans)
    index = numpy.searchsorted(self._indices, parent_index)
    if index == len(self._indices) or self._indices[index] != parent_index:
      raise ValueError
    else:
      return int(index), tail

class ReorderedTransforms(Transforms):
  '''A reordered :class:`Transforms` object.

  Parameters
  ----------
  parent : :class:`Transforms`
      The transforms to reorder.
  indices : one-dimensional array of :class:`int`\\s
      The new order of the transforms.
  '''

  __slots__ = '_parent', '_mask', '_indices'
  __cache__ = '_rindices'

  @types.apply_annotations
  def __init__(self, parent:stricttransforms, indices:types.arraydata):
    assert indices.dtype == int
    self._parent = parent
    self._indices = numpy.asarray(indices)
    super().__init__(parent.todims, parent.fromdim)

  @property
  def _rindices(self):
    return types.frozenarray(numpy.argsort(self._indices), copy=False)

  def __iter__(self):
    for itrans in self._indices:
      yield self._parent[int(itrans)]

  def __getitem__(self, index):
    if numeric.isintarray(index) and index.ndim == 1 and numpy.any(numpy.less(index, 0)):
      raise IndexError('index out of bounds')
    return self._parent[self._indices[index]]

  def __len__(self):
    return len(self._parent)

  def index_with_tail(self, trans):
    parent_index, tail = self._parent.index_with_tail(trans)
    return int(self._rindices[parent_index]), tail

class DerivedTransforms(Transforms):
  '''A sequence of derived transforms.

  The derived transforms are ordered first by parent transforms, then by derived
  transforms, as returned by the reference::

      (trans.append(ctrans) for trans, ref in zip(parent, parent_references) for ctrans in getattr(ref, derived_attribute))

  Parameters
  ----------
  parent : :class:`Transforms`
      The transforms to refine.
  parent_references: :class:`~nutils.elementseq.References`
      The references to use for the refinement.
  derived_attribute : :class:`str`
      The name of the attribute of a :class:`nutils.element.Reference` that
      contains the derived references.
  fromdim : :class:`int`
      The dimension all transforms in this sequence map from.
  '''

  __slots__ = '_parent', '_parent_references', '_derived_transforms'
  __cache__ = '_offsets'

  @types.apply_annotations
  def __init__(self, parent:stricttransforms, parent_references:types.strict[References], derived_attribute:types.strictstr, fromdim:types.strictint):
    if len(parent) != len(parent_references):
      raise ValueError('`parent` and `parent_references` should have the same length')
    if parent.fromdim != parent_references.ndims:
      raise ValueError('`parent` and `parent_references` have different dimensions')
    self._parent = parent
    self._parent_references = parent_references
    self._derived_transforms = operator.attrgetter(derived_attribute)
    super().__init__(parent.todims, fromdim)

  @property
  def _offsets(self):
    return types.frozenarray(numpy.cumsum([0, *(len(self._derived_transforms(ref)) for ref in self._parent_references)]), copy=False)

  def __len__(self):
    return self._offsets[-1]

  def __iter__(self):
    for reference, trans in zip(self._parent_references, self._parent):
      for dtrans in self._derived_transforms(reference):
        yield trans.append(dtrans)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    index = numeric.normdim(len(self), index)
    iparent = numpy.searchsorted(self._offsets, index, side='right')-1
    assert 0 <= iparent < len(self._offsets)-1
    iderived = index - self._offsets[iparent]
    return self._parent[iparent].append(self._derived_transforms(self._parent_references[iparent])[iderived])

  def index_with_tail(self, chains):
    iparent, parent_tails = self._parent.index_with_tail(chains)
    offset = self._offsets[iparent]
    for iderived, trans in enumerate(self._derived_transforms(self._parent_references[iparent])):
      try:
        return offset+iderived, parent_tails.remove_head(trans)
      except transform.HeadDoesNotMatch:
        continue
    raise ValueError

class UniformDerivedTransforms(Transforms):
  '''A sequence of refined transforms from a uniform sequence of references.

  The refined transforms are ordered first by parent transforms, then by
  derived transforms, as returned by the reference::

      (trans.append(ctrans) for trans in parent for ctrans in getattr(parent_reference, derived_attribute))

  Parameters
  ----------
  parent : :class:`Transforms`
      The transforms to refine.
  parent_reference: :class:`~nutils.element.Reference`
      The reference to use for the refinement.
  derived_attribute : :class:`str`
      The name of the attribute of a :class:`nutils.element.Reference` that
      contains the derived references.
  fromdim : :class:`int`
      The dimension all transforms in this sequence map from.
  '''

  __slots__ = '_parent', '_derived_transforms'

  @types.apply_annotations
  def __init__(self, parent:stricttransforms, parent_reference:element.strictreference, derived_attribute:types.strictstr, fromdim:types.strictint):
    if parent.fromdim != parent_reference.ndims:
      raise ValueError('`parent` and `parent_reference` have different dimensions')
    self._parent = parent
    self._derived_transforms = getattr(parent_reference, derived_attribute)
    super().__init__(parent.todims, fromdim)

  def __len__(self):
    return len(self._parent)*len(self._derived_transforms)

  def __iter__(self):
    for trans in self._parent:
      for dtrans in self._derived_transforms:
        yield trans.append(dtrans)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    iparent, iderived = divmod(numeric.normdim(len(self), index), len(self._derived_transforms))
    return self._parent[iparent].append(self._derived_transforms[iderived])

  def index_with_tail(self, chains):
    iparent, parent_tails = self._parent.index_with_tail(chains)
    offset = iparent*len(self._derived_transforms)
    for iderived, trans in enumerate(self._derived_transforms):
      try:
        return offset+iderived, parent_tails.remove_head(trans)
      except transform.HeadDoesNotMatch:
        continue
    raise ValueError

class ProductTransforms(Transforms):
  '''The product of two :class:`Transforms` objects.

  The order of the resulting transforms is: ``transforms1[0]*transforms2[0],
  transforms1[0]*transforms2[1], ..., transforms1[1]*transforms2[0],
  transforms1[1]*transforms2[1], ...``.

  Parameters
  ----------
  transforms1 : :class:`Transforms`
      The first sequence of transforms.
  transforms2 : :class:`Transforms`
      The second sequence of transforms.
  '''

  __slots__ = '_transforms1', '_transforms2'

  @types.apply_annotations
  def __init__(self, transforms1:stricttransforms, transforms2:stricttransforms):
    self._transforms1 = transforms1
    self._transforms2 = transforms2
    super().__init__(transforms1.todims+transforms2.todims, transforms1.fromdim+transforms2.fromdim)

  def __iter__(self):
    for chains1 in self._transforms1:
      for chains2 in self._transforms2:
        yield transform.TransformChains(*chains1, *chains2)

  def __getitem__(self, index):
    if not numeric.isint(index):
      return super().__getitem__(index)
    index1, index2 = divmod(numeric.normdim(len(self), index), len(self._transforms2))
    chain1 = transform.Bifurcate(self._transforms1[index1], self._transforms2[index2])
    return transform.TransformChains(*self._transforms1[index1], *self._transforms2[index2])

  def __len__(self):
    return len(self._transforms1) * len(self._transforms2)

  def index_with_tail(self, trans):
    assert len(trans) != len(self.todims)
    index1, tails1 = self._transforms1.index_with_tail(trans[:len(self._transforms1.todims)])
    index2, tails2 = self._transforms2.index_with_tail(trans[len(self._transforms1.todims):])
    return index1*len(self._transforms2)+index2, transform.TransformChains(*tails1, *tails2)

class ChainedTransforms(Transforms):
  '''A sequence of chained :class:`Transforms` objects.

  Parameters
  ----------
  items: :class:`tuple` of :class:`Transforms` objects
      The :class:`Transforms` objects to chain.
  '''

  __slots__ = '_items'
  __cache__ = '_offsets'

  @types.apply_annotations
  def __init__(self, items:types.tuple[stricttransforms]):
    if len(items) == 0:
      raise ValueError('Empty chain.')
    if len(set(item.todims for item in items)) != 1:
      raise ValueError('Cannot chain Transforms with different todims.')
    if len(set(item.fromdim for item in items)) != 1:
      raise ValueError('Cannot chain Transforms with different fromdim.')
    self._items = items
    super().__init__(self._items[0].todims, self._items[0].fromdim)

  @property
  def _offsets(self):
    return types.frozenarray(numpy.cumsum([0, *map(len, self._items)]), copy=False)

  def __len__(self):
    return self._offsets[-1]

  def __getitem__(self, index):
    if numeric.isint(index):
      index = numeric.normdim(len(self), index)
      outer = numpy.searchsorted(self._offsets, index, side='right') - 1
      assert outer >= 0 and outer < len(self._items)
      return self._items[outer][index-self._offsets[outer]]
    elif isinstance(index, slice) and index.step in (1, None):
      index = range(len(self))[index]
      if index == range(len(self)):
        return self
      elif index.start == index.stop:
        return EmptyTransforms(self.todims, self.fromdim)
      ostart = numpy.searchsorted(self._offsets, index.start, side='right') - 1
      ostop = numpy.searchsorted(self._offsets, index.stop, side='left')
      return chain((item[max(0,index.start-istart):min(istop-istart,index.stop-istart)] for item, (istart, istop) in zip(self._items[ostart:ostop], util.pairwise(self._offsets[ostart:ostop+1]))), self.todims, self.fromdim)
    elif numeric.isintarray(index) and index.ndim == 1 and len(index) and numpy.all(numpy.greater(numpy.diff(index), 0)):
      if index[0] < 0 or index[-1] >= len(self):
        raise IndexError('index out of bounds')
      split = numpy.searchsorted(index, self._offsets, side='left')
      return chain((item[index[start:stop]-offset] for item, offset, (start, stop) in zip(self._items, self._offsets, util.pairwise(split)) if stop > start), self.todims, self.fromdim)
    elif numeric.isboolarray(index) and index.shape == (len(self),):
      return chain((item[index[start:stop]] for item, (start, stop) in zip(self._items, util.pairwise(self._offsets))), self.todims, self.fromdim)
    else:
      return super().__getitem__(index)

  def __iter__(self):
    return itertools.chain.from_iterable(self._items)

  def index_with_tail(self, trans):
    offset = 0
    for item in self._items:
      try:
        index, tail = item.index_with_tail(trans)
        return index + offset, tail
      except ValueError:
        pass
      offset += len(item)
    raise ValueError

  def refined(self, references):
    return chain((item.refined(references[start:stop]) for item, start, stop in zip(self._items, self._offsets[:-1], self._offsets[1:])), self.todims, self.fromdim)

  def edges(self, references):
    return chain((item.edges(references[start:stop]) for item, start, stop in zip(self._items, self._offsets[:-1], self._offsets[1:])), self.todims, self.fromdim-1)

  def unchain(self):
    yield from self._items

def chain(items, todims, fromdim):
  '''Return the chained transforms sequence of ``items``.

  Parameters
  ----------
  items : iterable of :class:`Transforms` objects
      The :class:`Transforms` objects to chain.
  fromdim : :class:`int`
      The dimension all transforms in this sequence map from.

  Returns
  -------
  :class:`Transforms`
      The chained transforms.
  '''

  unchained = tuple(filter(len, itertools.chain.from_iterable(item.unchain() for item in items)))
  items_todims = set(item.todims for item in unchained)
  if not (items_todims <= {todims}):
    raise ValueError('expected transforms with todims={}, but got {}'.format(todims, items_todims))
  items_fromdim = set(item.fromdim for item in unchained)
  if not (items_fromdim <= {fromdim}):
    raise ValueError('expected transforms with fromdim={}, but got {}'.format(fromdim, items_fromdim))
  if len(unchained) == 0:
    return EmptyTransforms(todims, fromdim)
  elif len(unchained) == 1:
    return unchained[0]
  else:
    return ChainedTransforms(unchained)

class _EvaluableTransformChainsFromSequence(transform.EvaluableTransformChains):

  __slots__ = '_sequence'

  def __init__(self, sequence: Transforms, index: evaluable.Array) -> None:
    self._sequence = sequence
    super().__init__((index,), tuple(map(evaluable.Constant, sequence.todims)), (None,)*len(sequence.todims))

  def evalf(self, index: numpy.ndarray) -> transform.TransformChains:
    return self._sequence[index.__index__()]

class _EvaluableIndexWithTails(evaluable.Evaluable):

  __slots__ = '_sequence'

  def __init__(self, sequence: Transforms, chains: transform.EvaluableTransformChains) -> None:
    self._sequence = sequence
    super().__init__((chains,))

  def evalf(self, chains):
    index, tails = self._sequence.index_with_tail(chains)
    return numpy.array(index), tails

class _EvaluableTransformChainsFromTuple(transform.EvaluableTransformChains):

  __slots__ = '_index'

  def __init__(self, items: evaluable.Evaluable, index: int, todims: Tuple[Optional[evaluable.Array], ...], fromdims: Tuple[Optional[evaluable.Array], ...]) -> None:
    self._index = index
    super().__init__((items,), todims, fromdims)

  def evalf(self, items) -> numpy.ndarray:
    return items[self._index]

# vim:sw=2:sts=2:et
