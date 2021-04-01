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

"""
The transform module.
"""

from typing import Iterator, Optional, Union, Tuple, Dict, Iterable
from . import cache, numeric, util, types, warnings, evaluable
from .evaluable import Evaluable, Array
import numpy, collections, itertools, functools, operator
_ = numpy.newaxis

## EXCEPTIONS

class HeadDoesNotMatch(ValueError):
  '''The :class:`TransformChain` does not start with the given head :class:`TransformItem`.'''

## TRANSFORM CHAIN OPERATIONS

@types.lru_cache
def apply(chain, points):
  for trans in reversed(chain):
    points = trans.apply(points)
  return points

def n_ascending(chain):
  # number of ascending transform items counting from root (0). this is a
  # temporary hack required to deal with Bifurcate/Slice; as soon as we have
  # proper tensorial topologies we can switch back to strictly ascending
  # transformation chains.
  for n, trans in enumerate(chain):
    if trans.todim is not None and trans.todim < trans.fromdim:
      return n
  return len(chain)

def canonical(chain):
  # keep at lowest ndims possible; this is the required form for bisection
  n = n_ascending(chain)
  if n < 2:
    return tuple(chain)
  items = list(chain)
  i = 0
  while items[i].fromdim > items[n-1].fromdim:
    swapped = items[i+1].swapdown(items[i])
    if swapped:
      items[i:i+2] = swapped
      i -= i > 0
    else:
      i += 1
  return tuple(items)

def uppermost(chain):
  # bring to highest ndims possible
  n = n_ascending(chain)
  if n < 2:
    return tuple(chain)
  items = list(chain)
  i = n
  while items[i-1].todim < items[0].todim:
    swapped = items[i-2].swapup(items[i-1])
    if swapped:
      items[i-2:i] = swapped
      i += i < n
    else:
      i -= 1
  return tuple(items)

def promote(chain, ndims):
  # swap transformations such that ndims is reached as soon as possible, and
  # then maintained as long as possible (i.e. proceeds as canonical).
  for i, item in enumerate(chain): # NOTE possible efficiency gain using bisection
    if item.fromdim == ndims:
      return canonical(chain[:i+1]) + uppermost(chain[i+1:])
  return chain # NOTE at this point promotion essentially failed, maybe it's better to raise an exception

def linearfrom(chain, fromdim):
  todim = chain[0].todim if chain else fromdim
  while chain and fromdim < chain[-1].fromdim:
    chain = chain[:-1]
  if not chain:
    assert todim == fromdim
    return numpy.eye(fromdim)
  linear = numpy.eye(chain[-1].fromdim)
  for transitem in reversed(uppermost(chain)):
    linear = numpy.dot(transitem.linear, linear)
    if transitem.todim == transitem.fromdim + 1:
      linear = numpy.concatenate([linear, transitem.ext[:,_]], axis=1)
  assert linear.shape[0] == todim
  return linear[:,:fromdim] if linear.shape[1] >= fromdim \
    else numpy.concatenate([linear, numpy.zeros((todim, fromdim-linear.shape[1]))], axis=1)

## TRANSFORM ITEMS

class TransformItem(types.Singleton):
  '''Affine transformation.

  Base class for transformations of the type :math:`x ↦ A x + b`.

  Args
  ----
  todim : :class:`int`
      Dimension of the affine transformation domain.
  fromdim : :class:`int`
      Dimension of the affine transformation range.
  '''

  __slots__ = 'todim', 'fromdim'

  @types.apply_annotations
  def __init__(self, todim, fromdim:int):
    super().__init__()
    self.todim = todim
    self.fromdim = fromdim

  @property
  def todims(self):
    warnings.deprecation('`todims` has been renamed to `todim`')
    return self.todim

  @property
  def fromdims(self):
    warnings.deprecation('`fromdims` has been renamed to `fromdim`')
    return self.fromdim

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, self)

  def swapup(self, other):
    return None

  def swapdown(self, other):
    return None

  def factorize(self, todims):
    if sum(todims) != self.todim:
      raise ValueError('`todims` does not sum up to `self.todim`')
    i = 0
    while todims and todims[0] == 0:
      i += 1
      todims = todims[1:]
    j = 0
    while todims and todims[-1] == 0:
      j += 1
      todims = todims[:-1]
    if not todims:
      factorized = ()
    elif todims == (self.todim,):
      factorized = self,
    else:
      factorized = self.factorize_checked_and_trimmed(todims)
    return (Identity(0),)*i + factorized + (Identity(0),)*j

  def factorize_checked_and_trimmed(self, todims):
    if todims == (self.todim,):
      return self,
    else:
      raise ValueError('cannot factorize this transform item into the given todims')

stricttransformitem = types.strict[TransformItem]
stricttransform = types.tuple[stricttransformitem]

class Bifurcate(TransformItem):

  __slots__ = 'trans1', 'trans2'

  @types.apply_annotations
  def __init__(self, trans1:canonical, trans2:canonical):
    fromdim = trans1[-1].fromdim + trans2[-1].fromdim
    self.trans1 = trans1 + (Slice(0, trans1[-1].fromdim, fromdim),)
    self.trans2 = trans2 + (Slice(trans1[-1].fromdim, fromdim, fromdim),)
    super().__init__(todim=trans1[0].todim if trans1[0].todim == trans2[0].todim else None, fromdim=fromdim)

  def __str__(self):
    return '{}<>{}'.format(self.trans1, self.trans2)

  def apply(self, points):
    return apply(self.trans1, points), apply(self.trans2, points)

class Matrix(TransformItem):
  '''Affine transformation :math:`x ↦ A x + b`, with :math:`A` an :math:`n×m` matrix, :math:`n≥m`

  Parameters
  ----------
  linear : :class:`numpy.ndarray`
      The transformation matrix :math:`A`.
  offset : :class:`numpy.ndarray`
      The offset :math:`b`.
  '''

  __slots__ = 'linear', 'offset'

  @types.apply_annotations
  def __init__(self, linear:types.arraydata, offset:types.arraydata):
    assert linear.ndim == 2 and linear.dtype == float
    assert offset.ndim == 1 and offset.dtype == float
    assert offset.shape[0] == linear.shape[0]
    self.linear = numpy.asarray(linear)
    self.offset = numpy.asarray(offset)
    super().__init__(linear.shape[0], linear.shape[1])

  @types.lru_cache
  def apply(self, points):
    assert points.shape[-1] == self.fromdim
    return types.frozenarray(numpy.dot(points, self.linear.T) + self.offset, copy=False)

  def __mul__(self, other):
    assert isinstance(other, Matrix) and self.fromdim == other.todim
    linear = numpy.dot(self.linear, other.linear)
    offset = self.apply(other.offset)
    return Square(linear, offset) if self.todim == other.fromdim \
      else Updim(linear, offset, self.isflipped^other.isflipped) if self.todim == other.fromdim+1 \
      else Matrix(linear, offset)

  def __str__(self):
    if not hasattr(self, 'offset') or not hasattr(self, 'linear'):
      return '<uninitialized>'
    return util.obj2str(self.offset) + ''.join('+{}*x{}'.format(util.obj2str(v), i) for i, v in enumerate(self.linear.T))

class Square(Matrix):
  '''Affine transformation :math:`x ↦ A x + b`, with :math:`A` square

  Parameters
  ----------
  linear : :class:`numpy.ndarray`
      The transformation matrix :math:`A`.
  offset : :class:`numpy.ndarray`
      The offset :math:`b`.
  '''

  __slots__ = '_transform_matrix',
  __cache__ ='det',

  @types.apply_annotations
  def __init__(self, linear:types.arraydata, offset:types.arraydata):
    assert linear.shape[0] == linear.shape[1]
    self._transform_matrix = {}
    super().__init__(linear, offset)

  @types.lru_cache
  def invapply(self, points):
    return types.frozenarray(numpy.linalg.solve(self.linear, (points - self.offset).T).T, copy=False)

  @property
  def det(self):
    return numpy.linalg.det(self.linear)

  @property
  def isflipped(self):
    return self.fromdim > 0 and self.det < 0

  @types.lru_cache
  def transform_poly(self, coeffs):
    assert coeffs.ndim == self.fromdim + 1
    degree = coeffs.shape[1] - 1
    assert all(n == degree+1 for n in coeffs.shape[2:])
    try:
      M = self._transform_matrix[degree]
    except KeyError:
      eye = numpy.eye(self.fromdim, dtype=int)
      # construct polynomials for affine transforms of individual dimensions
      polys = numpy.zeros((self.fromdim,)+(2,)*self.fromdim)
      polys[(slice(None),)+(0,)*self.fromdim] = self.offset
      for idim, e in enumerate(eye):
        polys[(slice(None),)+tuple(e)] = self.linear[:,idim]
      # reduces polynomials to smallest nonzero power
      polys = [poly[tuple(slice(None if p else 1) for p in poly[tuple(eye)])] for poly in polys]
      # construct transform poly by transforming all monomials separately and summing
      M = numpy.zeros((degree+1,)*(2*self.fromdim), dtype=float)
      for powers in numpy.ndindex(*[degree+1]*self.fromdim):
        if sum(powers) <= degree:
          M_power = functools.reduce(numeric.poly_mul, [numeric.poly_pow(poly, power) for poly, power in zip(polys, powers)])
          M[tuple(slice(n) for n in M_power.shape)+powers] += M_power
      self._transform_matrix[degree] = M
    return types.frozenarray(numpy.einsum('jk,ik', M.reshape([(degree+1)**self.fromdim]*2), coeffs.reshape(coeffs.shape[0],-1)).reshape(coeffs.shape), copy=False)

class Shift(Square):
  '''Shift transformation :math:`x ↦ x + b`

  Parameters
  ----------
  offset : :class:`numpy.ndarray`
      The offset :math:`b`.
  '''

  __slots__ = ()

  det = 1.

  @types.apply_annotations
  def __init__(self, offset:types.arraydata):
    assert offset.ndim == 1 and offset.dtype == float
    super().__init__(numpy.eye(offset.shape[0]), offset)

  @types.lru_cache
  def apply(self, points):
    return types.frozenarray(points + self.offset, copy=False)

  @types.lru_cache
  def invapply(self, points):
    return types.frozenarray(points - self.offset, copy=False)

  def __str__(self):
    return '{}+x'.format(util.obj2str(self.offset))

class Identity(Shift):
  '''Identity transformation :math:`x ↦ x`

  Parameters
  ----------
  ndims : :class:`int`
      Dimension of :math:`x`.
  '''

  __slots__ = ()

  def __init__(self, ndims):
    super().__init__(numpy.zeros(ndims))

  def apply(self, points):
    return points

  def invapply(self, points):
    return points

  def __str__(self):
    return 'x'

  def factorize_checked_and_trimmed(self, todims):
    return tuple(map(Identity, todims))

class Scale(Square):
  '''Affine transformation :math:`x ↦ a x + b`, with :math:`a` a scalar

  Parameters
  ----------
  scale : :class:`float`
      The scalar :math:`a`.
  offset : :class:`numpy.ndarray`
      The offset :math:`b`.
  '''

  __slots__ = 'scale',

  @types.apply_annotations
  def __init__(self, scale:float, offset:types.arraydata):
    assert offset.ndim == 1 and offset.dtype == float
    self.scale = scale
    super().__init__(numpy.eye(offset.shape[0]) * scale, offset)

  @types.lru_cache
  def apply(self, points):
    return types.frozenarray(self.scale * points + self.offset, copy=False)

  @types.lru_cache
  def invapply(self, points):
    return types.frozenarray((points - self.offset) / self.scale, copy=False)

  @property
  def det(self):
    return self.scale**self.todim

  def __str__(self):
    return '{}+{}*x'.format(util.obj2str(self.offset), self.scale)

  def __mul__(self, other):
    assert isinstance(other, Matrix) and self.fromdim == other.todim
    if isinstance(other, Scale):
      return Scale(self.scale * other.scale, self.apply(other.offset))
    return super().__mul__(other)

class Updim(Matrix):
  '''Affine transformation :math:`x ↦ A x + b`, with :math:`A` an :math:`n×(n-1)` matrix

  Parameters
  ----------
  linear : :class:`numpy.ndarray`
      The transformation matrix :math:`A`.
  offset : :class:`numpy.ndarray`
      The offset :math:`b`.
  '''

  __slots__ = 'isflipped',
  __cache__ = 'ext',

  @types.apply_annotations
  def __init__(self, linear:types.arraydata, offset:types.arraydata, isflipped:bool):
    assert linear.shape[0] == linear.shape[1] + 1
    self.isflipped = isflipped
    super().__init__(linear, offset)

  @property
  def ext(self):
    ext = numeric.ext(self.linear)
    return types.frozenarray(-ext if self.isflipped else ext, copy=False)

  @property
  def flipped(self):
    return Updim(self.linear, self.offset, not self.isflipped)

  def swapdown(self, other):
    if isinstance(other, TensorChild):
      return ScaledUpdim(other, self), Identity(self.fromdim)

class SimplexEdge(Updim):

  __slots__ = 'iedge', 'inverted'

  swap = (
    ((1,0), (2,0), (3,0), (7,1)),
    ((0,1), (2,1), (3,1), (6,1)),
    ((0,2), (1,2), (3,2), (5,1)),
    ((0,3), (1,3), (2,3), (4,3)),
  )

  @types.apply_annotations
  def __init__(self, ndims:types.strictint, iedge:types.strictint, inverted:bool=False):
    assert ndims >= iedge >= 0
    self.iedge = iedge
    self.inverted = inverted
    vertices = numpy.concatenate([numpy.zeros(ndims)[_,:], numpy.eye(ndims)], axis=0)
    coords = vertices[list(range(iedge))+list(range(iedge+1,ndims+1))]
    super().__init__((coords[1:]-coords[0]).T, coords[0], inverted^(iedge%2))

  @property
  def flipped(self):
    return SimplexEdge(self.todim, self.iedge, not self.inverted)

  def swapup(self, other):
    # prioritize ascending transformations, i.e. change updim << scale to scale << updim
    if isinstance(other, SimplexChild):
      ichild, iedge = self.swap[self.iedge][other.ichild]
      return SimplexChild(self.todim, ichild), SimplexEdge(self.todim, iedge, self.inverted)
    elif self.fromdim == 0 and other == Identity(0):
      ichild, iedge = self.swap[self.iedge][0]
      return SimplexChild(self.todim, ichild), SimplexEdge(self.todim, iedge, self.inverted)

  def swapdown(self, other):
    # prioritize decending transformations, i.e. change scale << updim to updim << scale
    if isinstance(other, SimplexChild):
      key = other.ichild, self.iedge
      for iedge, children in enumerate(self.swap[:self.todim+1]):
        try:
          ichild = children[:2**self.fromdim].index(key)
        except ValueError:
          pass
        else:
          return SimplexEdge(self.todim, iedge, self.inverted), SimplexChild(self.fromdim, ichild) if self.fromdim else Identity(0)

class SimplexChild(Square):

  __slots__ = 'ichild',

  def __init__(self, ndims, ichild):
    if ndims == 0:
      raise ValueError('Cannot create a 0D `SimplexChild`, use `Identity(0)` instead.')
    self.ichild = ichild
    if ichild <= ndims:
      linear = numpy.eye(ndims) * .5
      offset = linear[ichild-1] if ichild else numpy.zeros(ndims)
    elif ndims == 2 and ichild == 3:
      linear = (-.5,0), (.5,.5)
      offset = .5, 0
    elif ndims == 3 and ichild == 4:
      linear = (-.5,0,-.5), (.5,.5,0), (0,0,.5)
      offset = .5, 0, 0
    elif ndims == 3 and ichild == 5:
      linear = (0,-.5,0), (.5,0,0), (0,.5,.5)
      offset = .5, 0, 0
    elif ndims == 3 and ichild == 6:
      linear = (.5,0,0), (0,-.5,0), (0,.5,.5)
      offset = 0, .5, 0
    elif ndims == 3 and ichild == 7:
      linear = (-.5,0,-.5), (-.5,-.5,0), (.5,.5,.5)
      offset = .5, .5, 0
    else:
      raise NotImplementedError('SimplexChild(ndims={}, ichild={})'.format(ndims, ichild))
    super().__init__(linear, offset)

class Slice(Matrix):

  __slots__ = 's',

  @types.apply_annotations
  def __init__(self, i1:int, i2:int, fromdim:int):
    todim = i2-i1
    assert 0 <= todim <= fromdim
    self.s = slice(i1,i2)
    super().__init__(numpy.eye(fromdim)[self.s], numpy.zeros(todim))

  def apply(self, points):
    return types.frozenarray(points[:,self.s])

class ScaledUpdim(Updim):

  __slots__ = 'trans1', 'trans2'

  def __init__(self, trans1, trans2):
    assert trans1.todim == trans1.fromdim == trans2.todim == trans2.fromdim + 1
    self.trans1 = trans1
    self.trans2 = trans2
    super().__init__(numpy.dot(trans1.linear, trans2.linear), trans1.apply(trans2.offset), trans1.isflipped^trans2.isflipped)

  def swapup(self, other):
    if type(other) is Identity:
      return self.trans1, self.trans2

  @property
  def flipped(self):
    return ScaledUpdim(self.trans1, self.trans2.flipped)

class TensorEdge1(Updim):

  __slots__ = 'trans',

  def __init__(self, trans1, ndims2):
    self.trans = trans1
    super().__init__(linear=numeric.blockdiag([trans1.linear, numpy.eye(ndims2)]), offset=numpy.concatenate([trans1.offset, numpy.zeros(ndims2)]), isflipped=trans1.isflipped)

  def swapup(self, other):
    # prioritize ascending transformations, i.e. change updim << scale to scale << updim
    if isinstance(other, TensorChild) and self.trans.fromdim == other.trans1.todim:
      swapped = self.trans.swapup(other.trans1)
      trans2 = other.trans2
    elif isinstance(other, (TensorChild, SimplexChild)) and other.fromdim == other.todim and not self.trans.fromdim:
      swapped = self.trans.swapup(Identity(0))
      trans2 = other
    else:
      swapped = None
    if swapped:
      child, edge = swapped
      return TensorChild(child, trans2), TensorEdge1(edge, trans2.fromdim)

  def swapdown(self, other):
    # prioritize ascending transformations, i.e. change scale << updim to updim << scale
    if isinstance(other, TensorChild) and other.trans1.fromdim == self.trans.todim:
      swapped = self.trans.swapdown(other.trans1)
      if swapped:
        edge, child = swapped
        return TensorEdge1(edge, other.trans2.todim), TensorChild(child, other.trans2) if child.fromdim else other.trans2
      return ScaledUpdim(other, self), Identity(self.fromdim)

  @property
  def flipped(self):
    return TensorEdge1(self.trans.flipped, self.fromdim-self.trans.fromdim)

  def factorize_checked_and_trimmed(self, todims):
    for i in range(len(todims)+1):
      if sum(todims[:i]) == self.trans.todim:
        return self.trans.factorize_checked_and_trimmed(todims[:i]) + tuple(map(Identity, todims[i:]))
    raise ValueError('cannot factorize this transform item into the given todims')

class TensorEdge2(Updim):

  __slots__ = 'trans'

  def __init__(self, ndims1, trans2):
    self.trans = trans2
    super().__init__(linear=numeric.blockdiag([numpy.eye(ndims1), trans2.linear]), offset=numpy.concatenate([numpy.zeros(ndims1), trans2.offset]), isflipped=trans2.isflipped^(ndims1%2))

  def swapup(self, other):
    # prioritize ascending transformations, i.e. change updim << scale to scale << updim
    if isinstance(other, TensorChild) and self.trans.fromdim == other.trans2.todim:
      swapped = self.trans.swapup(other.trans2)
      trans1 = other.trans1
    elif isinstance(other, (TensorChild, SimplexChild)) and other.fromdim == other.todim and not self.trans.fromdim:
      swapped = self.trans.swapup(Identity(0))
      trans1 = other
    else:
      swapped = None
    if swapped:
      child, edge = swapped
      return TensorChild(trans1, child), TensorEdge2(trans1.fromdim, edge)

  def swapdown(self, other):
    # prioritize ascending transformations, i.e. change scale << updim to updim << scale
    if isinstance(other, TensorChild) and other.trans2.fromdim == self.trans.todim:
      swapped = self.trans.swapdown(other.trans2)
      if swapped:
        edge, child = swapped
        return TensorEdge2(other.trans1.todim, edge), TensorChild(other.trans1, child) if child.fromdim else other.trans1
      return ScaledUpdim(other, self), Identity(self.fromdim)

  @property
  def flipped(self):
    return TensorEdge2(self.fromdim-self.trans.fromdim, self.trans.flipped)

  def factorize_checked_and_trimmed(self, todims):
    for i in reversed(range(len(todims)+1)):
      if sum(todims[i:]) == self.trans.todim:
        return tuple(map(Identity, todims[:i])) + self.trans.factorize_checked_and_trimmed(todims[i:])
    raise ValueError('cannot factorize this transform item into the given todims')

class TensorChild(Square):

  __slots__ = 'trans1', 'trans2'
  __cache__ = 'det',

  def __init__(self, trans1, trans2):
    assert trans1.fromdim and trans2.fromdim
    self.trans1 = trans1
    self.trans2 = trans2
    linear = numeric.blockdiag([trans1.linear, trans2.linear])
    offset = numpy.concatenate([trans1.offset, trans2.offset])
    super().__init__(linear, offset)

  @property
  def det(self):
    return self.trans1.det * self.trans2.det

  def factorize_checked_and_trimmed(self, todims):
    for i in range(len(todims)+1):
      if sum(todims[i:]) == self.trans1.todim:
        j = i
        while j < len(todims) and todims[j] == 0:
          j += 1
        return self.trans1.factorize_checked_and_trimmed(todims[:i]) + (Identity(0),)*(j-i) + self.trans2.factorize_checked_and_trimmed(todims[j:])
    raise ValueError('cannot factorize this transform item into the given todims')

class Identifier(Identity):
  '''Generic identifier

  This transformation serves as an element-specific or topology-specific token
  to form the basis of transformation lookups. Otherwise, the transform behaves
  like an identity.
  '''

  __slots__ = 'token'

  @types.apply_annotations
  def __init__(self, ndims:int, token):
    self.token = token
    super().__init__(ndims)

  def __str__(self):
    return ':'.join(map(str, self._args))

## TRANSFORM CHAIN

class TransformChain:
  '''A chain of :class:`TransformItem`.

  Parameters
  ----------
  *items : :class:`TransformItem`
      The list of transform items that comprises the chain.
  todim : :class:`int`, optional
      The dimension the transform chain maps to. If the transform chain is
      empty, this argument must be supplied.

  Attributes
  ----------
  todim : :class:`int`
      The dimension the transform chain maps to.
  fromdim : :class:`int`
      The dimension the transform chain maps from.
  '''

  __slots__ = '_items', 'todim', 'fromdim'

  @classmethod
  def empty(cls, todim: int) -> 'TransformChain':
    '''Create an empty chain with the given dimension.'''
    return cls(todim=todim)

  def __init__(self, *items: TransformItem, todim: Optional[int] = None) -> None:
    self._items = tuple(items)
    assert all(isinstance(item, TransformItem) for item in self._items)
    if self._items:
      self.todim = self._items[0].todim
      if todim is not None and self._items[0].todim != todim:
        raise ValueError('The `todim` of the first item in this chain does not match the given `todim`.')
    elif todim is not None:
      self.todim = todim
    else:
      raise ValueError('An empty `TransformChain` must be initialized with a `todim`.')
    self.fromdim = self._items[-1].fromdim if self._items else self.todim

  def __len__(self) -> int:
    return len(self._items)

  def get_item(self, index: int) -> TransformItem:
    return self._items[index]

  def __iter__(self) -> Iterator[TransformItem]:
    return iter(self._items)

  def __reversed__(self) -> Iterator[TransformItem]:
    return reversed(self._items)

  def __getitem__(self, index):
    if isinstance(index, slice):
      start, stop, step = index.indices(len(self._items))
      if step != 1:
        raise IndexError
      if start < len(self._items):
        return TransformChain(*self._items[start:stop], todim=self._items[start].todim)
      else:
        return TransformChain(todim=self.fromdim)
    elif isinstance(index, int):
      return self._items[index]
    else:
      raise IndexError

  def __eq__(self, other) -> bool:
    return type(self) == type(other) and self._items == other._items and self.todim == other.todim

  def __hash__(self) -> int:
    return hash((self._items, self.todim))

  def __repr__(self) -> str:
    return 'TransformChain({})'.format(', '.join(map(repr, self._items))) if self._items else 'TransformChain(todim={})'.format(self.todim)

  def extend(self, chain: 'TransformChain') -> 'TransformChain':
    if self.fromdim != chain.todim:
      raise ValueError('Cannot append {} with todim {} to this chain with fromdim {}.'.format(chain, chain.todim, self.fromdim))
    return TransformChain(*self._items, *chain._items, todim=self.todim)

  def append(self, item: TransformItem) -> 'TransformChain':
    '''Append a :class:`TransformItem` to this chain.'''
    if self.fromdim != item.todim:
      raise ValueError('Cannot append {} with todim {} to this chain with fromdim {}.'.format(item, item.todim, self.fromdim))
    return TransformChain(*self._items, item, todim=self.todim)

  def remove_head(self, head: TransformItem) -> 'TransformChain':
    '''Remove the given head from this chain.

    Parameters
    ----------
    head : :class:`TransformItem`
        The transform item to remove from this chain.

    Returns
    -------
    tail : :class:`TransformChain`
        The remainder of this chain after removing the head.

    Raises
    ------
    HeadDoesNotMatch
        If this chain does not start with the given head.
    '''

    if head.todim != self.todim:
      raise ValueError('Expected a `TransformItem` with todim {} but got {}.'.format(self.todim, head.todim))
    if head.todim == head.fromdim:
      items = self.uppermost._items
    else:
      items = self.canonical._items
    if items and items[0] == head:
      return TransformChain(*items[1:], todim=head.fromdim)
    else:
      raise HeadDoesNotMatch('{} does not start with {}'.format(self, head))

  @types.lru_cache
  def apply(self, points: numpy.ndarray) -> numpy.ndarray:
    for trans in reversed(self._items):
      points = trans.apply(points)
    return points

  @property
  def _n_ascending(self) -> int:
    # number of ascending transform items counting from root (0). this is a
    # temporary hack required to deal with Bifurcate/Slice; as soon as we have
    # proper tensorial topologies we can switch back to strictly ascending
    # transformation chains.
    for n, trans in enumerate(self._items):
      if trans.todim is not None and trans.todim < trans.fromdim:
        return n
    return len(self._items)

  @property
  def canonical(self) -> 'TransformChain':
    # keep at lowest ndims possible; this is the required form for bisection
    n = self._n_ascending
    if n < 2:
      return self
    items = list(self._items)
    i = 0
    while items[i].fromdim > items[n-1].fromdim:
      swapped = items[i+1].swapdown(items[i])
      if swapped:
        items[i:i+2] = swapped
        i -= i > 0
      else:
        i += 1
    return TransformChain(*items, todim=self.todim)

  @property
  def uppermost(self) -> 'TransformChain':
    # bring to highest ndims possible
    n = self._n_ascending
    if n < 2:
      return self
    items = list(self._items)
    i = n
    while items[i-1].todim < items[0].todim:
      swapped = items[i-2].swapup(items[i-1])
      if swapped:
        items[i-2:i] = swapped
        i += i < n
      else:
        i -= 1
    return TransformChain(*items, todim=self.todim)

  def promote(self, ndims: int) -> 'TransformChain':
    # swap transformations such that ndims is reached as soon as possible, and
    # then maintained as long as possible (i.e. proceeds as canonical).
    for i, item in enumerate(self._items): # NOTE possible efficiency gain using bisection
      if item.fromdim == ndims:
        return self[:i+1].canonical.extend(self[i+1:].uppermost)
    return self # NOTE at this point promotion essentially failed, maybe it's better to raise an exception

  def linearfrom(self, fromdim: int) -> numpy.ndarray:
    while self and fromdim < self.fromdim:
      self = self[:-1]
    if not self:
      assert self.fromdim == fromdim
      return numpy.eye(fromdim)
    linear = numpy.eye(self.fromdim)
    for transitem in reversed(self.uppermost._items):
      linear = numpy.dot(transitem.linear, linear)
      if transitem.todim == transitem.fromdim + 1:
        linear = numpy.concatenate([linear, transitem.ext[:,_]], axis=1)
    assert linear.shape[0] == self.todim
    return linear[:,:fromdim] if linear.shape[1] >= fromdim \
      else numpy.concatenate([linear, numpy.zeros((self.todim, fromdim-linear.shape[1]))], axis=1)

  @property
  def linear(self) -> numpy.ndarray:
    linear = numpy.eye(self.fromdim)
    for item in reversed(self._items):
      linear = numpy.dot(item.linear, linear)
    assert linear.shape == (self.todim, self.fromdim)
    return linear

  @property
  def extended_linear(self) -> numpy.ndarray:
    linear = numpy.eye(self.fromdim)
    for item in reversed(self._items):
      linear = numpy.dot(item.linear, linear)
      assert item.fromdim <= item.todim <= item.fromdim + 1
      if item.todim == item.fromdim + 1:
        linear = numpy.concatenate([linear, item.ext[:,_]], axis=1)
    assert linear.shape == (self.todim, self.todim)
    return linear

class TransformChains:
  '''A list of chains of :class:`TransformItem`.

  Parameters
  ----------
  *chains : :class:`TransformChain`
      The list of transform chains.

  Attributes
  ----------
  nchains : :class:`int`
      The number of individual transform chains.
  todims : :class:`tuple` of :class:`int`
      The dimensions the individual transform chains map to.
  todim : :class:`int`
      The sum of dimensions the individual transform chains maps to.
  fromdims : :class:`tuple` of :class:`int`
      The dimensions the individual transform chains map from.
  fromdim : :class:`int`
      The sum of dimensions the individual transform chains map from.
  '''

  __slots__ = 'nchains', 'todims', 'todim', 'fromdim', 'fromdims', '_chains', '_toslices', '_fromslices'

  @classmethod
  def empty(cls, *todims: int) -> 'TransformChains':
    '''Create empty chains with the given dimensions.'''
    return cls(*map(TransformChain.empty, todims))

  def __init__(self, *chains: TransformChain) -> None:
    assert all(isinstance(chain, TransformChain) for chain in chains)
    self.nchains = len(chains)
    self._chains = tuple(chains)
    self.todims = tuple(chain.todim for chain in chains)
    self.todim = sum(self.todims)
    self.fromdims = tuple(chain.fromdim for chain in chains)
    self.fromdim = sum(self.fromdims)
    self._toslices = tuple(slice(l, r) for l, r in util.pairwise(util.cumsum(self.todims+(0,))))
    self._fromslices = tuple(slice(l, r) for l, r in util.pairwise(util.cumsum(self.fromdims+(0,))))

  def __len__(self) -> int:
    return len(self._chains)

  def get_chain(self, index: int) -> TransformChain:
    return self._chains[index]

  def __iter__(self) -> Iterator[TransformChain]:
    return iter(self._chains)

  def __reversed__(self) -> Iterator[TransformChain]:
    return reversed(self._chains)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return TransformChains(*self._chains[index])
    elif isinstance(index, int):
      return self._chains[index]
    else:
      raise IndexError

  def __eq__(self, other) -> bool:
    return type(self) == type(other) and self._chains == other._chains

  def __hash__(self) -> int:
    return hash(self._chains)

  def __repr__(self) -> str:
    return 'TransformChains({})'.format(', '.join(map(repr, self._chains)))

  def apply(self, points: numpy.ndarray) -> numpy.ndarray:
    if points.ndim == 0:
      raise ValueError('Expected an array with at least 1 dimension but got 0.')
    if points.shape[-1] != self.fromdim:
      raise ValueError('The last axis of `points` must have length {} but got {}.'.format(self.fromdim, points.shape[-1]))
    result = numpy.concatenate([chain.apply(points[...,s]) for chain, s in zip(self._chains, self._fromslices)], axis=-1)
    assert result.shape[-1] == self.todim
    return result

  def append(self, item: TransformItem) -> 'TransformChains':
    '''Append a :class:`TransformItem` to this list of chains.'''
    if item.todim != self.fromdim:
      raise ValueError('Expected a `TransformItem` with todim={} but got {}.'.format(self.fromdim, item.todim))
    return TransformChains(*(chain.append(t) for chain, t in zip(self._chains, item.factorize(self.fromdims))))

  def remove_head(self, head: TransformItem) -> 'TransformChains':
    '''Remove the given head from this list of chains.

    Parameters
    ----------
    head : :class:`TransformItem`
        The transform item to remove from this list of chains.

    Returns
    -------
    tail : :class:`TransformChains`
        The remainder of the chains after removing the head.

    Raises
    ------
    HeadDoesNotMatch
        If this list of chains does not start with the given head.
    '''

    if head.todim != self.todim:
      raise ValueError('Expected a `TransformItem` with todim {} but got {}.'.format(self.todim, head.todim))
    return TransformChains(*(chain.remove_head(h) for chain, h in zip(self._chains, head.factorize(self.todims))))

  @property
  def canonical(self) -> 'TransformChains':
    return TransformChains(*(chain.canonical for chain in self._chains))

  @property
  def uppermost(self) -> 'TransformChains':
    return TransformChains(*(chain.uppermost for chain in self._chains))

  def linearfrom(self, fromdim: int) -> numpy.ndarray:
    if len(self._chains) != 1:
      raise NotImplementedError
    return self._chains[0].linearfrom(fromdim)

  @property
  def linear(self) -> numpy.ndarray:
    linear = numpy.zeros((self.todim, self.fromdim), dtype=float)
    for chain, s, t in zip(self._chains, self._toslices, self._fromslices):
      linear[s, t] = chain.linear
    return linear

  @property
  def extended_linear(self) -> numpy.ndarray:
    linear = numpy.zeros((self.todim, self.todim), dtype=float)
    for chain, s in zip(self._chains, self._toslices):
      linear[s, s] = chain.extended_linear
    return linear

## EVALUABLE TRANSFORM CHAIN

def _as_optional_evaluable_dim(dim: Optional[Union[int, Array]]) -> Optional[Array]:
  if dim is None:
    return None
  elif isinstance(dim, int):
    if dim < 0:
      raise ValueError('expected a non-negative integer but got {}'.format(dim))
    return evaluable.Constant(dim)
  elif isinstance(dim, Array):
    if dim.dtype != int or dim.ndim != 0:
      raise ValueError('expected a scalar integer array but got {}'.format(dim))
    return dim
  else:
    raise ValueError('expected None, a non-negative integer or a scalar integer array but got {}'.format(dim))

def _check_optional_evaluable_dim(dim: Optional[Array]) -> None:
  if not (dim is None or isinstance(dim, Array) and dim.ndim == 0 and dim.dtype == int):
    raise ValueError('expected None or a scalar integer array but got {}'.format(dim))

class EvaluableTransformChain(Evaluable):
  'The :class:`~nutils.evaluable.Evaluable` equivalent of a :class:`TransformChain`.'

  __slots__ = '_todim', '_fromdim'

  @staticmethod
  def from_argument(name: str, todim: Optional[Union[int, Array]] = None, fromdim: Optional[Union[int, Array]] = None) -> 'EvaluableTransformChain':
    todim = _as_optional_evaluable_dim(todim)
    fromdim = _as_optional_evaluable_dim(fromdim)
    return _TransformChainArgument(name, todim, fromdim)

  def __init__(self, args: Tuple[Evaluable, ...], todim: Optional[Array], fromdim: Optional[Array]) -> None:
    _check_optional_evaluable_dim(todim)
    _check_optional_evaluable_dim(fromdim)
    self._todim = todim
    self._fromdim = fromdim
    super().__init__(args)

  @property
  def todim(self) -> Array:
    if self._todim is None:
      return _ToDim(self)
    else:
      return self._todim

  @property
  def fromdim(self) -> Array:
    if self._fromdim is None:
      return _FromDim(self)
    else:
      return self._fromdim

  @property
  def linear(self) -> Array:
    return _Linear(self)

  @property
  def extended_linear(self) -> Array:
    return _ExtendedLinear(self)

  def apply(self, coords: Array) -> Array:
    return _Apply(self, coords)

class EvaluableTransformChains(Evaluable):
  'The :class:`~nutils.evaluable.Evaluable` equivalent of a :class:`TransformChains`.'

  __slots__ = 'nchains', '_todims', '_fromdims'

  @staticmethod
  def from_argument(name: str, todims: Iterable[Optional[Union[int, Array]]], fromdims: Optional[Iterable[Optional[Union[int, Array]]]] = None) -> 'EvaluableTransformChains':
    evaluable_todims = tuple(map(_as_optional_evaluable_dim, todims))
    if fromdims is None:
      evaluable_fromdims = (None,)*len(evaluable_todims) # type: Tuple[Optional[Array], ...]
    else:
      evaluable_fromdims = tuple(map(_as_optional_evaluable_dim, fromdims))
    return _TransformChainsArgument(name, evaluable_todims, evaluable_fromdims)

  @staticmethod
  def from_individual_chains(*chains: EvaluableTransformChain) -> 'EvaluableTransformChains':
    return _JoinTransformChains(*chains)

  def __init__(self, args: Tuple[Evaluable, ...], todims: Tuple[Optional[Array], ...], fromdims: Tuple[Optional[Array], ...]) -> None:
    if len(todims) != len(fromdims):
      raise ValueError('the lengths of `todims` and `fromdims` differ')
    for todim in todims:
      _check_optional_evaluable_dim(todim)
    for fromdim in fromdims:
      _check_optional_evaluable_dim(fromdim)
    self.nchains = len(todims)
    self._todims = todims
    self._fromdims = fromdims
    super().__init__(args)

  def __len__(self) -> int:
    return self.nchains

  def get_chain(self, index: int) -> EvaluableTransformChain:
    index = numeric.normdim(self.nchains, index)
    return _GetTransformChain(self, index, self._todims[index], self._fromdims[index])

  def __getitem__(self, index: int) -> EvaluableTransformChain:
    return self.get_chain(index)

  def __iter__(self) -> Iterator[EvaluableTransformChain]:
    return map(self.get_chain, range(self.nchains))

  def __reversed__(self) -> Iterator[EvaluableTransformChain]:
    return map(self.get_chain, reversed(range(self.nchains)))

  @property
  def todims(self) -> Tuple[Array, ...]:
    return tuple(self.get_chain(index).todim for index in range(self.nchains))

  @property
  def todim(self) -> Array:
    return util.sum(self.todims, evaluable.zeros((), int))

  @property
  def fromdims(self) -> Tuple[Array, ...]:
    return tuple(self.get_chain(index).fromdim for index in range(self.nchains))

  @property
  def fromdim(self) -> Array:
    return util.sum(self.fromdims, evaluable.zeros((), int))

  @property
  def linear(self) -> Array:
    return _Linear(self)

  @property
  def extended_linear(self) -> Array:
    return _ExtendedLinear(self)

  def apply(self, coords: Array) -> Array:
    return _Apply(self, coords)

class _ToDim(Array):

  __slots__ = ()

  def __init__(self, chain: Union[EvaluableTransformChain, EvaluableTransformChains]) -> None:
    super().__init__(args=(chain,), shape=(), dtype=int)

  def evalf(self, chain: Union[TransformChain, TransformChains]) -> numpy.ndarray:
    return numpy.array(chain.todim)

class _FromDim(Array):

  __slots__ = ()

  def __init__(self, chain: Union[EvaluableTransformChain, EvaluableTransformChains]) -> None:
    super().__init__(args=(chain,), shape=(), dtype=int)

  def evalf(self, chain: Union[TransformChain, TransformChains]) -> numpy.ndarray:
    return numpy.array(chain.fromdim)

class _Linear(Array):

  __slots__ = ()

  def __init__(self, chain: Union[EvaluableTransformChain, EvaluableTransformChains]) -> None:
    super().__init__(args=(chain,), shape=(chain.todim, chain.fromdim), dtype=float)

  def evalf(self, chain: Union[TransformChain, TransformChains]) -> numpy.ndarray:
    return chain.linear

  def _derivative(self, var: evaluable.DerivativeTargetBase, seen: Dict[Evaluable, Evaluable]) -> Array:
    return evaluable.zeros(self.shape+var.shape, dtype=float)

class _ExtendedLinear(Array):

  __slots__ = ()

  def __init__(self, chain: Union[EvaluableTransformChain, EvaluableTransformChains]) -> None:
    super().__init__(args=(chain,), shape=(chain.todim, chain.todim), dtype=float)

  def evalf(self, chain: Union[TransformChain, TransformChains]) -> numpy.ndarray:
    return chain.extended_linear

  def _derivative(self, var: evaluable.DerivativeTargetBase, seen: Dict[Evaluable, Evaluable]) -> Array:
    return evaluable.zeros(self.shape+var.shape, dtype=float)

class _Apply(Array):

  __slots__ = '_chain', '_points'

  def __init__(self, chain: Union[EvaluableTransformChain, EvaluableTransformChains], points: Array) -> None:
    if points.ndim == 0:
      raise ValueError('expected a points array with at least one axis but got {}'.format(points))
    if not evaluable.equalindex(chain.fromdim, points.shape[-1]):
      raise ValueError('the last axis of points does not match the from dimension of the transform chain')
    self._chain = chain
    self._points = points
    super().__init__(args=(chain, points), shape=(*points.shape[:-1], chain.todim), dtype=float)

  def evalf(self, chain: Union[TransformChain, TransformChains], points: numpy.ndarray) -> numpy.ndarray:
    return chain.apply(points)

  def _derivative(self, var: evaluable.DerivativeTargetBase, seen: Dict[Evaluable, Evaluable]) -> Array:
    axis = self._points.ndim - 1
    linear = evaluable.appendaxes(evaluable.prependaxes(self._chain.linear, self._points.shape[:-1]), var.shape)
    dpoints = evaluable.insertaxis(evaluable.derivative(self._points, var, seen), axis, linear.shape[axis])
    return evaluable.dot(linear, dpoints, axis+1)

class _TransformChainArgument(EvaluableTransformChain):

  __slots__ = '_name'

  def __init__(self, name: str, todim: Optional[Array], fromdim: Optional[Array]) -> None:
    self._name = name
    super().__init__((evaluable.EVALARGS,), todim, fromdim)

  def evalf(self, evalargs) -> TransformChain:
    chain = evalargs[self._name]
    assert isinstance(chain, TransformChain)
    return chain

  @property
  def arguments(self):
    return frozenset({self})

class _TransformChainsArgument(EvaluableTransformChains):

  __slots__ = '_name'

  def __init__(self, name: str, todims: Tuple[Optional[Array], ...], fromdims: Tuple[Optional[Array], ...]) -> None:
    self._name = name
    super().__init__((evaluable.EVALARGS,), todims, fromdims)

  def evalf(self, evalargs) -> TransformChains:
    chains = evalargs[self._name]
    assert isinstance(chains, TransformChains)
    return chains

  @property
  def arguments(self):
    return frozenset({self})

class _JoinTransformChains(EvaluableTransformChains):

  __slots__ = '_chains'

  def __init__(self, *chains: EvaluableTransformChain) -> None:
    self._chains = chains
    super().__init__(chains, tuple(chain.todim for chain in chains), tuple(chain.fromdim for chain in chains))

  def get_chain(self, index: int) -> EvaluableTransformChain:
    return self._chains[index]

  def evalf(self, *chains: TransformChain) -> TransformChains:
    return TransformChains(*chains)

class _GetTransformChain(EvaluableTransformChain):

  __slots__ = '_index'

  def __init__(self, chains: EvaluableTransformChains, index: int, todim: Optional[Array], fromdim: Optional[Array]) -> None:
    assert 0 <= index < len(chains)
    self._index = index
    super().__init__((chains,), todim=todim, fromdim=fromdim)

  def evalf(self, chains: TransformChains) -> TransformChain:
    return chains.get_chain(self._index)

# vim:sw=2:sts=2:et
