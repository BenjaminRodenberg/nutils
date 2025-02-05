import numpy, itertools, pickle, warnings as _builtin_warnings, operator
from nutils import evaluable, function, mesh, numeric, types, points, transformseq, transform, element
from nutils.testing import *
_ = numpy.newaxis


class Array(TestCase):

  def test_cast_ndim_mismatch(self):
    with self.assertRaises(ValueError):
      function.Array.cast([1,2], ndim=2)

  def test_cast_dtype_mismatch(self):
    with self.assertRaises(ValueError):
      function.Array.cast([1.2,2.3], dtype=int)

  def test_cast_invalid_argument(self):
    with self.assertRaises(ValueError):
      function.Array.cast('132')

  def test_ndim(self):
    self.assertEqual(function.Argument('a', (2,3)).ndim, 2)

  def test_shape(self):
    n = function.Argument('n', (), dtype=int)
    l1, l2, l3 = function.Argument('a', (2, function.Array.cast(3),n)).shape
    with self.subTest('int'):
      self.assertEqual(l1, 2)
    with self.subTest('const Array'):
      self.assertEqual(l2, 3)
    with self.subTest('unknown'):
      self.assertEqual(l3.prepare_eval(npoints=None).simplified, n.prepare_eval(npoints=None).simplified)

  def test_size_known(self):
    self.assertEqual(function.Argument('a', (2,3)).size, 6)

  def test_size_0d(self):
    self.assertEqual(function.Argument('a', ()).size, 1)

  def test_size_unknown(self):
    n = function.Argument('n', (), dtype=int)
    size = function.Argument('a', (2,n,3)).size
    self.assertIsInstance(size, function.Array)
    self.assertEqual(size.prepare_eval(npoints=None).simplified, (2*n*3).prepare_eval(npoints=None).simplified)

  def test_len_0d(self):
    with self.assertRaisesRegex(Exception, '^len\\(\\) of unsized object$'):
      len(function.Array.cast(0))

  def test_len_known(self):
    self.assertEqual(len(function.Array.cast([1,2])), 2)

  def test_len_unknown(self):
    with self.assertRaisesRegex(Exception, '^unknown length$'):
      len(function.Argument('a', [function.Argument('b', (), dtype=int)]))

  def test_iter_0d(self):
    with self.assertRaisesRegex(Exception, '^iteration over a 0-D array$'):
      iter(function.Array.cast(0))

  def test_iter_known(self):
    a, b = function.Array.cast([1,2])
    self.assertEqual(a.prepare_eval(npoints=None).eval(), 1)
    self.assertEqual(b.prepare_eval(npoints=None).eval(), 2)

  def test_iter_unknown(self):
    with self.assertRaisesRegex(Exception, '^iteration over array with unknown length$'):
      iter(function.Argument('a', [function.Argument('b', (), dtype=int)]))

  def test_binop_notimplemented(self):
    with self.assertRaises(TypeError):
      function.Argument('a', ()) + '1'

  def test_rbinop_notimplemented(self):
    with self.assertRaises(TypeError):
      '1' + function.Argument('a', ())

  def test_deprecated_simplified(self):
    with self.assertWarns(warnings.NutilsDeprecationWarning):
      function.Array.cast([1,2]).simplified


class integral_compatibility(TestCase):

  def test_eval(self):
    v = numpy.array([1,2])
    a = function.Argument('a', (2,), dtype=float)
    self.assertAllAlmostEqual(a.eval(a=v), v)

  def test_derivative(self):
    v = numpy.array([1,2])
    a = function.Argument('a', (2,), dtype=float)
    f = 2*a.sum()
    for name, obj in ('str', 'a'), ('argument', a):
      with self.subTest(name):
        self.assertAllAlmostEqual(f.derivative(obj).eval(a=v), numpy.array([2,2]))

  def test_derivative_str_evaluable_shape(self):
    a = function.Argument('a', (), dtype=int)
    b = function.Argument('b', (a,), dtype=float)
    f = b**2
    with self.assertRaises(ValueError):
      f.derivative('b')

  def test_derivative_str_unknown_argument(self):
    f = function.zeros((2,), dtype=float)
    with self.assertRaises(ValueError):
      f.derivative('a')

  def test_derivative_invalid(self):
    f = function.zeros((2,), dtype=float)
    with self.assertRaises(ValueError):
      f.derivative(1.)

  def test_replace(self):
    v = numpy.array([1,2])
    a = function.Argument('a', (2,), dtype=int)
    b = function.Argument('b', (2,), dtype=int)
    f = a.replace(dict(a=b))
    self.assertAllAlmostEqual(f.eval(b=v), v)

  def test_contains(self):
    f = 2*function.Argument('a', (2,), dtype=int)
    self.assertTrue(f.contains('a'))
    self.assertFalse(f.contains('b'))

  def test_argshapes(self):
    a = function.Argument('a', (2,3), dtype=int)
    b = function.Argument('b', (3,), dtype=int)
    f = (a * b[None]).sum(-1)
    self.assertEqual(dict(f.argshapes), dict(a=(2,3), b=(3,)))

  def test_argshapes_shape_mismatch(self):
    f = function.Argument('a', (2,), dtype=int)[None] + function.Argument('a', (3,), dtype=int)[:,None]
    with self.assertRaises(Exception):
      f.argshapes

  def test_argshapes_evaluable_shape(self):
    a = function.Argument('a', (), dtype=int)
    b = function.Argument('b', (a,), dtype=float)
    f = b**2
    with self.assertRaises(ValueError):
      f.argshapes

@parametrize
class check(TestCase):

  def setUp(self):
    super().setUp()
    numpy.random.seed(0)

  def assertArrayAlmostEqual(self, actual, desired, decimal):
    if actual.shape != desired.shape:
      self.fail('shapes of actual {} and desired {} are incompatible.'.format(actual.shape, desired.shape))
    error = actual - desired if not actual.dtype.kind == desired.dtype.kind == 'b' else actual ^ desired
    approx = error.dtype.kind in 'fc'
    mask = numpy.greater_equal(abs(error), 1.5 * 10**-decimal) if approx else error
    indices = tuple(zip(*mask.nonzero())) if actual.ndim else ((),) if mask.any() else ()
    if not indices:
      return
    lines = ['arrays are not equal']
    if approx:
      lines.append(' up to {} decimals'.format(decimal))
    lines.append(' in {}/{} entries:'.format(len(indices), error.size))
    n = 5
    lines.extend('\n  {} actual={} desired={} difference={}'.format(index, actual[index], desired[index], error[index]) for index in indices[:n])
    if len(indices) > 2*n:
      lines.append('\n  ...')
      n = -n
    lines.extend('\n  {} actual={} desired={} difference={}'.format(index, actual[index], desired[index], error[index]) for index in indices[n:])
    self.fail(''.join(lines))

  def test_lower_eval(self):
    args = tuple((numpy.random.randint if self.dtype == int else numpy.random.uniform)(size=shape, low=self.low, high=self.high) for shape in self.shapes)
    actual = self.op(*args).prepare_eval(npoints=None).eval()
    desired = self.n_op(*args)
    self.assertArrayAlmostEqual(actual, desired, decimal=15)

def _check(name, op, n_op, shapes, low=None, high=None, dtype=float):
  if low is None:
    low = 0 if dtype == int else -1
  if high is None:
    high = 20 if dtype == int else 1
  check(name, op=op, n_op=n_op, shapes=shapes, low=low, high=high, dtype=dtype)

_check('asarray', function.asarray, lambda a: a, [(2,4,2)])
_check('zeros', lambda: function.zeros([1,4,3,4]), lambda: numpy.zeros([1,4,3,4]), [])
_check('ones', lambda: function.ones([1,4,3,4]), lambda: numpy.ones([1,4,3,4]), [])
_check('eye', lambda: function.eye(3), lambda: numpy.eye(3), [])

_check('add', function.add, numpy.add, [(4,), (4,)])
_check('Array_add', lambda a, b: function.Array.cast(a) + b, numpy.add, [(4,), (4,)])
_check('Array_radd', lambda a, b: a + function.Array.cast(b), numpy.add, [(4,), (4,)])
_check('subtract', function.subtract, numpy.subtract, [(4,), (4,)])
_check('Array_sub', lambda a, b: function.Array.cast(a) - b, numpy.subtract, [(4,), (4,)])
_check('Array_rsub', lambda a, b: a - function.Array.cast(b), numpy.subtract, [(4,), (4,)])
_check('negative', function.negative, numpy.negative, [(4,)])
_check('Array_neg', lambda a: -function.Array.cast(a), numpy.negative, [(4,)])
_check('Array_pos', lambda a: +function.Array.cast(a), lambda a: a, [(4,)])
_check('multiply', function.multiply, numpy.multiply, [(4,), (4,)])
_check('Array_mul', lambda a, b: function.Array.cast(a) * b, numpy.multiply, [(4,), (4,)])
_check('Array_rmul', lambda a, b: a * function.Array.cast(b), numpy.multiply, [(4,), (4,)])
_check('divide', function.divide, numpy.divide, [(4,), (4,)])
_check('Array_truediv', lambda a, b: function.Array.cast(a) / b, numpy.divide, [(4,), (4,)])
_check('Array_rtruediv', lambda a, b: a / function.Array.cast(b), numpy.divide, [(4,), (4,)])
_check('floor_divide', function.floor_divide, numpy.floor_divide, [(4,4,), (4,)], dtype=int, low=1, high=10)
_check('Array_floordiv', lambda a, b: function.Array.cast(a) // b, numpy.floor_divide, [(4,4,), (4,)], dtype=int, low=1, high=10)
_check('Array_rfloordiv', lambda a, b: a // function.Array.cast(b), numpy.floor_divide, [(4,4,), (4,)], dtype=int, low=1, high=10)
_check('reciprocal', function.reciprocal, numpy.reciprocal, [(4,)])
_check('power', function.power, numpy.power, [(4,), (4,)], low=1, high=2)
_check('Array_pow', lambda a, b: function.Array.cast(a) ** b, numpy.power, [(4,), (4,)], low=1, high=2)
_check('Array_rpow', lambda a, b: a ** function.Array.cast(b), numpy.power, [(4,), (4,)], low=1, high=2)
_check('sqrt', function.sqrt, numpy.sqrt, [(4,)])
_check('abs', function.abs, numpy.abs, [(4,)])
_check('Array_abs', lambda a: abs(function.Array.cast(a)), numpy.abs, [(4,)])
_check('sign', function.sign, numpy.sign, [(4,)])
_check('mod', function.mod, numpy.mod, [(4,4), (4,)], dtype=int, low=1, high=10)
_check('Array_mod', lambda a, b: function.Array.cast(a) % b, numpy.mod, [(4,4), (4,)], dtype=int, low=1, high=10)
_check('Array_rmod', lambda a, b: a % function.Array.cast(b), numpy.mod, [(4,4), (4,)], dtype=int, low=1, high=10)

_check('cos', function.cos, numpy.cos, [(4,)])
_check('sin', function.sin, numpy.sin, [(4,)])
_check('tan', function.tan, numpy.tan, [(4,)])
_check('arccos', function.arccos, numpy.arccos, [(4,)])
_check('arcsin', function.arcsin, numpy.arcsin, [(4,)])
_check('arctan', function.arctan, numpy.arctan, [(4,)])
_check('arctan2', function.arctan2, numpy.arctan2, [(4,1),(1,4)])
_check('cosh', function.cosh, numpy.cosh, [(4,)])
_check('sinh', function.sinh, numpy.sinh, [(4,)])
_check('tanh', function.tanh, numpy.tanh, [(4,)])
_check('arctanh', function.arctanh, numpy.arctanh, [(4,)])
_check('exp', function.exp, numpy.exp, [(4,)])
_check('log', function.log, numpy.log, [(4,)], low=0)
_check('log2', function.log2, numpy.log2, [(4,)], low=0)
_check('log10', function.log10, numpy.log10, [(4,)], low=0)

_check('greater', function.greater, numpy.greater, [(4,1),(1,4)])
_check('equal', function.equal, numpy.equal, [(4,1),(1,4)])
_check('less', function.less, numpy.less, [(4,1),(1,4)])
_check('min', function.min, numpy.minimum, [(4,4),(4,4)])
_check('max', function.max, numpy.maximum, [(4,4),(4,4)])
_check('heaviside', function.heaviside, lambda u: numpy.heaviside(u, .5), [(4,4)])

# TODO: opposite
# TODO: mean
# TODO: jump

_check('sum', lambda a: function.sum(a,2), lambda a: a.sum(2), [(4,3,4)])
_check('Array_sum', lambda a: function.Array.cast(a).sum(2), lambda a: a.sum(2), [(4,3,4)])
_check('product', lambda a: function.product(a,2), lambda a: numpy.product(a,2), [(4,3,4)])
_check('Array_prod', lambda a: function.Array.cast(a).prod(2), lambda a: numpy.product(a,2), [(4,3,4)])

_check('dot', lambda a,b: function.dot(a,b,axes=2), lambda a,b: (a*b).sum(2), [(4,2,4),(4,2,4)])
_check('Array_dot', lambda a,b: function.Array.cast(a).dot(b,axes=2), lambda a,b: (a*b).sum(2), [(4,2,4),(4,2,4)])
_check('trace', function.trace, numpy.trace, [(3,3)])
_check('norm2', function.norm2, lambda a: numpy.linalg.norm(a, axis=1), [(2,3)])
_check('normalized', function.normalized, lambda a: a / numpy.linalg.norm(a, axis=1)[:,None], [(2,3)])
_check('Array_normalized', lambda a: function.Array.cast(a).normalized(), lambda a: a / numpy.linalg.norm(a, axis=1)[:,None], [(2,3)])
_check('inverse', lambda a: function.inverse(a+3*numpy.eye(3)), lambda a: numpy.linalg.inv(a+3*numpy.eye(3)), [(2,3,3)])
_check('determinant', lambda a: function.determinant(a+3*numpy.eye(3)), lambda a: numpy.linalg.det(a+3*numpy.eye(3)), [(2,3,3)])
_check('eigval', lambda a: function.eig(a)[0], lambda a: numpy.diag(numpy.linalg.eig(a)[0]), [(3,3)])
_check('eigvec', lambda a: function.eig(a)[1], lambda a: numpy.linalg.eig(a)[1], [(3,3)])
_check('eigval_symmetric', lambda a: function.eig(a+a.T)[0], lambda a: numpy.diag(numpy.linalg.eig(a+a.T)[0]), [(3,3)])
_check('eigvec_symmetric', lambda a: function.eig(a+a.T)[1], lambda a: numpy.linalg.eig(a+a.T)[1], [(3,3)])
_check('takediag', function.takediag, numpy.diag, [(3,3)])
_check('diagonalize', function.diagonalize, numpy.diag, [(3,)])
_check('cross', function.cross, numpy.cross, [(3,), (3,)])
_check('outer', function.outer, lambda a, b: a[:,None]*b[None,:], [(2,3),(4,3)])
_check('outer_self', function.outer, lambda a: a[:,None]*a[None,:], [(2,3)])

_check('transpose', lambda a: function.transpose(a,[0,1,3,2]), lambda a: a.transpose([0,1,3,2]), [(1,2,3,4)], dtype=int)
_check('Array_transpose', lambda a: function.Array.cast(a).transpose([0,1,3,2]), lambda a: a.transpose([0,1,3,2]), [(1,2,3,4)], dtype=int)
_check('insertaxis', lambda a: function.insertaxis(a,2,3), lambda a: numpy.repeat(numpy.expand_dims(a,2), 3, 2), [(3,2,4)], dtype=int)
_check('expand_dims', lambda a: function.expand_dims(a,1), lambda a: numpy.expand_dims(a,1), [(2,3)], dtype=int)
_check('repeat', lambda a: function.repeat(a,3,1), lambda a: numpy.repeat(a,3,1), [(2,1,4)], dtype=int)
_check('swapaxes', lambda a: function.swapaxes(a,1,2), lambda a: numpy.transpose(a, (0,2,1)), [(2,3,4)], dtype=int)
_check('Array_swapaxes', lambda a: function.Array.cast(a).swapaxes(1,2), lambda a: numpy.transpose(a, (0,2,1)), [(2,3,4)], dtype=int)
_check('ravel', lambda a: function.ravel(a,1), lambda a: numpy.reshape(a, (2,12,5)), [(2,3,4,5)], dtype=int)
_check('unravel', lambda a: function.unravel(a,1,(3,4)), lambda a: numpy.reshape(a, (2,3,4,5)), [(2,12,5)], dtype=int)
_check('take', lambda a: function.take(a, numpy.array([[0,2],[1,3]]), 1), lambda a: numpy.take(a, numpy.array([[0,2],[1,3]]), 1), [(3,4,5)], dtype=int)
_check('take_bool', lambda a: function.take(a, numpy.array([False, True, False, True]), 1), lambda a: numpy.compress(numpy.array([False, True, False, True]), a, 1), [(3,4,5)], dtype=int)
_check('get', lambda a: function.take(a, 1, 1), lambda a: numpy.take(a, 1, 1), [(3,4,5)], dtype=int)
_check('inflate', lambda a: function.inflate(a,numpy.array([0,3]), 4, 1), lambda a: numpy.concatenate([a[:,:1], numpy.zeros_like(a), a[:,1:]], axis=1), [(4,2,4)])
_check('kronecker', lambda a: function.kronecker(a,1,3,1), lambda a: numpy.stack([numpy.zeros_like(a), a, numpy.zeros_like(a)], axis=1), [(4,4)])
_check('concatenate', lambda a, b: function.concatenate([a,b], axis=1), lambda a, b: numpy.concatenate([a,b], axis=1), [(4,2,1),(4,3,1)])
_check('stack', lambda a,b: function.stack([a,b], 1), lambda a,b: numpy.stack([a,b], 1), [(4,2),(4,2)])

_check('Array_getitem_scalar', lambda a: function.Array.cast(a)[0], lambda a: a[0], [(5,3,2)], dtype=int)
_check('Array_getitem_scalar_scalar', lambda a: function.Array.cast(a)[0,1], lambda a: a[0,1], [(5,3,2)], dtype=int)
_check('Array_getitem_slice_step', lambda a: function.Array.cast(a)[:,::2], lambda a: a[:,::2], [(5,3,2)], dtype=int)
_check('Array_getitem_ellipsis_scalar', lambda a: function.Array.cast(a)[...,1], lambda a: a[...,1], [(5,3,2)], dtype=int)
_check('Array_getitem_ellipsis_scalar_newaxis', lambda a: function.Array.cast(a)[...,1,None], lambda a: a[...,1,None], [(5,3,2)], dtype=int)

_check('add_T', lambda a: function.add_T(a, (1, 2)), lambda a: a + a.transpose((0,2,1)), [(5,2,2)], dtype=int)
_check('Array_add_T', lambda a: function.Array.cast(a).add_T((1, 2)), lambda a: a + a.transpose((0,2,1)), [(5,2,2)], dtype=int)


class broadcasting(TestCase):

  def test_singleton_expansion(self):
    a = function.Argument('a', (1,2,3))
    b = function.Argument('b', (3,1,3))
    c = function.Argument('b', (3,1,1))
    (a_, b_, c_), shape, dtype = function._broadcast(a, b, c)
    self.assertEqual(shape, (3,2,3))
    self.assertEqual(a_.shape, (3,2,3))
    self.assertEqual(b_.shape, (3,2,3))
    self.assertEqual(c_.shape, (3,2,3))

  def test_prepend_axes(self):
    a = function.Argument('a', (3,2,3))
    b = function.Argument('b', (3,))
    c = function.Argument('b', (2,3))
    (a_, b_, c_), shape, dtype = function._broadcast(a, b, c)
    self.assertEqual(shape, (3,2,3))
    self.assertEqual(a_.shape, (3,2,3))
    self.assertEqual(b_.shape, (3,2,3))
    self.assertEqual(c_.shape, (3,2,3))

  def test_both(self):
    a = function.Argument('a', (3,2,3))
    b = function.Argument('b', (3,))
    c = function.Argument('b', (1,1))
    (a_, b_, c_), shape, dtype = function._broadcast(a, b, c)
    self.assertEqual(shape, (3,2,3))
    self.assertEqual(a_.shape, (3,2,3))
    self.assertEqual(b_.shape, (3,2,3))
    self.assertEqual(c_.shape, (3,2,3))

  def test_incompatible_shape(self):
    a = function.Argument('a', (3,2,3))
    b = function.Argument('b', (4,))
    with self.assertRaisesRegex(Exception, 'incompatible lengths'):
      function._broadcast(a, b)

  def test_runtime_check(self):
    n = function.Argument('n', (), dtype=int)
    m = function.Argument('m', (), dtype=int)
    a = function.Argument('a', (n,), dtype=int)
    b = function.Argument('b', (m,), dtype=int)
    (a_, b_), shape, dtype = function._broadcast(a, b)
    with self.subTest('match'):
      self.assertEqual(shape[0].prepare_eval(npoints=None).eval(n=numpy.array(2), m=numpy.array(2)), 2)
    with self.subTest('mismatch'):
      with self.assertRaises(evaluable.EvaluationError):
        shape[0].prepare_eval(npoints=None).eval(n=numpy.array(2), m=numpy.array(3))


@parametrize
class sampled(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, geom = mesh.unitsquare(4, self.etype)
    basis = self.domain.basis('std', degree=1)
    numpy.random.seed(0)
    self.f = basis.dot(numpy.random.uniform(size=len(basis)))
    sample = self.domain.sample('gauss', 2)
    self.f_sampled = sample.asfunction(sample.eval(self.f))

  def test_isarray(self):
    self.assertTrue(function.isarray(self.f_sampled))

  def test_values(self):
    diff = self.domain.integrate(self.f - self.f_sampled, ischeme='gauss2')
    self.assertAllAlmostEqual(diff, 0)

  def test_pointset(self):
    with self.assertRaises(evaluable.EvaluationError):
      self.domain.integrate(self.f_sampled, ischeme='uniform2')

for etype in 'square', 'triangle', 'mixed':
  sampled(etype=etype)


@parametrize
class piecewise(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.rectilinear([1])
    x, = self.geom
    if self.partition:
      left, mid, right = function.partition(x, .2, .8)
      self.f = left + function.sin(x) * mid + x**2 * right
    else:
      self.f = function.piecewise(x, [.2,.8], 1, function.sin(x), x**2)

  def test_evalf(self):
    f_ = self.domain.sample('uniform', 4).eval(self.f) # x=.125, .375, .625, .875
    assert numpy.equal(f_, [1, numpy.sin(.375), numpy.sin(.625), .875**2]).all()

  def test_deriv(self):
    g_ = self.domain.sample('uniform', 4).eval(function.grad(self.f, self.geom)) # x=.125, .375, .625, .875
    assert numpy.equal(g_, [[0], [numpy.cos(.375)], [numpy.cos(.625)], [2*.875]]).all()

piecewise(partition=False)
piecewise(partition=True)


class elemwise(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, geom = mesh.rectilinear([5])
    self.index = self.domain.f_index
    self.data = tuple(map(types.frozenarray, (
      numpy.arange(1, dtype=float).reshape(1,1),
      numpy.arange(2, dtype=float).reshape(1,2),
      numpy.arange(3, dtype=float).reshape(3,1),
      numpy.arange(4, dtype=float).reshape(2,2),
      numpy.arange(6, dtype=float).reshape(3,2),
    )))
    self.func = function.Elemwise(self.data, self.index, float)

  def test_evalf(self):
    for i, trans in enumerate(self.domain.transforms):
      with self.subTest(i=i):
        numpy.testing.assert_array_almost_equal(self.func.prepare_eval(ndims=self.domain.ndims).eval(_transforms=(trans,), _points=points.SimplexGaussPoints(self.domain.ndims, 1)), self.data[i][_])

  def test_shape(self):
    for i, trans in enumerate(self.domain.transforms):
      with self.subTest(i=i):
        self.assertEqual(self.func.size.prepare_eval(ndims=self.domain.ndims, npoints=None).eval(_transforms=(trans,)), self.data[i].size)

  def test_derivative(self):
    self.assertTrue(evaluable.iszero(function.localgradient(self.func, self.domain.ndims).prepare_eval(ndims=self.domain.ndims)))

  def test_shape_derivative(self):
    self.assertEqual(function.localgradient(self.func, self.domain.ndims).shape, self.func.shape+(self.domain.ndims,))


class replace_arguments(TestCase):

  def test_array(self):
    a = function.Argument('a', (2,))
    b = function.Array.cast([1,2])
    self.assertEqual(function.replace_arguments(a, dict(a=b)).prepare_eval(npoints=None), b.prepare_eval(npoints=None))

  def test_argument(self):
    a = function.Argument('a', (2,))
    b = function.Argument('b', (2,))
    self.assertEqual(function.replace_arguments(a, dict(a=b)).prepare_eval(npoints=None), b.prepare_eval(npoints=None))

  def test_argument_array(self):
    a = function.Argument('a', (2,))
    b = function.Argument('b', (2,))
    c = function.Array.cast([1,2])
    self.assertEqual(function.replace_arguments(function.replace_arguments(a, dict(a=b)), dict(b=c)).prepare_eval(npoints=None), c.prepare_eval(npoints=None))

  def test_swap(self):
    a = function.Argument('a', (2,))
    b = function.Argument('b', (2,))
    self.assertEqual(function.replace_arguments(2*a+3*b, dict(a=b, b=a)).prepare_eval(npoints=None), (2*b+3*a).prepare_eval(npoints=None))

  def test_ignore_replaced(self):
    a = function.Argument('a', (2,))
    b = function.Array.cast([1,2])
    c = function.Array.cast([2,3])
    self.assertEqual(function.replace_arguments(function.replace_arguments(a, dict(a=b)), dict(a=c)).prepare_eval(npoints=None), b.prepare_eval(npoints=None))

  def test_ignore_recursion(self):
    a = function.Argument('a', (2,))
    self.assertEqual(function.replace_arguments(a, dict(a=2*a)).prepare_eval(npoints=None), (2*a).prepare_eval(npoints=None))

  def test_replace_derivative(self):
    a = function.Argument('a', ())
    b = function.Argument('b', ())
    self.assertEqual(function.replace_arguments(function.derivative(a, a), dict(a=b)).prepare_eval(npoints=None).simplified, evaluable.ones(()).simplified)


class namespace(TestCase):

  def test_set_scalar(self):
    ns = function.Namespace()
    ns.scalar = 1

  def test_set_array(self):
    ns = function.Namespace()
    ns.array = function.zeros([2,3])

  def test_set_scalar_expression(self):
    ns = function.Namespace()
    ns.scalar = '1'

  def test_set_array_expression(self):
    ns = function.Namespace()
    ns.foo = function.zeros([3,3])
    ns.array_ij = 'foo_ij + foo_ji'

  def test_set_readonly(self):
    ns = function.Namespace()
    with self.assertRaises(AttributeError):
      ns._foo = None

  def test_set_readonly_internal(self):
    ns = function.Namespace()
    with self.assertRaises(AttributeError):
      ns._attributes = None

  def test_del_existing(self):
    ns = function.Namespace()
    ns.foo = function.zeros([2,3])
    del ns.foo

  def test_del_readonly_internal(self):
    ns = function.Namespace()
    with self.assertRaises(AttributeError):
      del ns._attributes

  def test_del_nonexisting(self):
    ns = function.Namespace()
    with self.assertRaises(AttributeError):
      del ns.foo

  def test_get_nonexisting(self):
    ns = function.Namespace()
    with self.assertRaises(AttributeError):
      ns.foo

  def test_invalid_default_geometry_no_str(self):
    with self.assertRaises(ValueError):
      function.Namespace(default_geometry_name=None)

  def test_invalid_default_geometry_no_variable(self):
    with self.assertRaises(ValueError):
      function.Namespace(default_geometry_name='foo_bar')

  def assertEqualLowered(self, actual, desired, **lowerargs):
    return self.assertEqual(actual.prepare_eval(**lowerargs), desired.prepare_eval(**lowerargs))

  def test_default_geometry_property(self):
    ns = function.Namespace()
    ns.x = 1
    self.assertEqualLowered(ns.default_geometry, ns.x, ndims=0)
    ns = function.Namespace(default_geometry_name='y')
    ns.y = 2
    self.assertEqualLowered(ns.default_geometry, ns.y, ndims=0)

  def test_copy(self):
    ns = function.Namespace()
    ns.foo = function.zeros([2,3])
    ns = ns.copy_()
    self.assertTrue(hasattr(ns, 'foo'))

  def test_copy_change_geom(self):
    ns1 = function.Namespace()
    domain, ns1.y = mesh.rectilinear([2,2])
    ns1.basis = domain.basis('spline', degree=2)
    ns2 = ns1.copy_(default_geometry_name='y')
    self.assertEqual(ns2.default_geometry_name, 'y')
    self.assertEqualLowered(ns2.eval_ni('basis_n,i'), ns2.basis.grad(ns2.y), ndims=domain.ndims)

  def test_copy_preserve_geom(self):
    ns1 = function.Namespace(default_geometry_name='y')
    domain, ns1.y = mesh.rectilinear([2,2])
    ns1.basis = domain.basis('spline', degree=2)
    ns2 = ns1.copy_()
    self.assertEqual(ns2.default_geometry_name, 'y')
    self.assertEqualLowered(ns2.eval_ni('basis_n,i'), ns2.basis.grad(ns2.y), ndims=domain.ndims)

  def test_copy_fixed_lengths(self):
    ns = function.Namespace(length_i=2)
    ns = ns.copy_()
    self.assertEqual(ns.eval_ij('δ_ij').shape, (2,2))

  def test_copy_fallback_length(self):
    ns = function.Namespace(fallback_length=2)
    ns = ns.copy_()
    self.assertEqual(ns.eval_ij('δ_ij').shape, (2,2))

  def test_eval(self):
    ns = function.Namespace()
    ns.foo = function.zeros([3,3])
    ns.eval_ij('foo_ij + foo_ji')

  def test_eval_fixed_lengths(self):
    ns = function.Namespace(length_i=2)
    self.assertEqual(ns.eval_ij('δ_ij').shape, (2,2))

  def test_eval_fixed_lengths_multiple(self):
    ns = function.Namespace(length_jk=2)
    self.assertEqual(ns.eval_ij('δ_ij').shape, (2,2))
    self.assertEqual(ns.eval_ik('δ_ik').shape, (2,2))

  def test_eval_fallback_length(self):
    ns = function.Namespace(fallback_length=2)
    self.assertEqual(ns.eval_ij('δ_ij').shape, (2,2))

  def test_matmul_0d(self):
    ns = function.Namespace()
    ns.foo = 2
    self.assertEqualLowered('foo' @ ns, ns.foo, npoints=None)

  def test_matmul_1d(self):
    ns = function.Namespace()
    ns.foo = function.zeros([2])
    self.assertEqualLowered('foo_i' @ ns, ns.foo, npoints=None)

  def test_matmul_2d(self):
    ns = function.Namespace()
    ns.foo = function.zeros([2, 3])
    with self.assertRaises(ValueError):
      'foo_ij' @ ns

  def test_matmul_nostr(self):
    ns = function.Namespace()
    with self.assertRaises(TypeError):
      1 @ ns

  def test_matmul_fixed_lengths(self):
    ns = function.Namespace(length_i=2)
    self.assertEqual(('1_i δ_ij' @ ns).shape, (2,))

  def test_matmul_fallback_length(self):
    ns = function.Namespace(fallback_length=2)
    self.assertEqual(('1_i δ_ij' @ ns).shape, (2,))

  def test_replace(self):
    ns = function.Namespace(default_geometry_name='y')
    ns.foo = function.Argument('arg', [2,3])
    ns.bar_ij = 'sin(foo_ij) + cos(2 foo_ij)'
    ns = ns(arg=function.zeros([2,3]))
    self.assertEqualLowered(ns.foo, function.zeros([2,3]), npoints=None)
    self.assertEqual(ns.default_geometry_name, 'y')

  def test_pickle(self):
    orig = function.Namespace()
    domain, geom = mesh.unitsquare(2, 'square')
    orig.x = geom
    orig.v = function.stack([1, geom[0], geom[0]**2], 0)
    orig.u = 'v_n ?lhs_n'
    orig.f = 'cosh(x_0)'
    pickled = pickle.loads(pickle.dumps(orig))
    for attr in ('x', 'v', 'u', 'f'):
      self.assertEqualLowered(getattr(pickled, attr), getattr(orig, attr), ndims=domain.ndims)
    self.assertEqual(pickled.arg_shapes['lhs'], orig.arg_shapes['lhs'])

  def test_pickle_default_geometry_name(self):
    orig = function.Namespace(default_geometry_name='g')
    pickled = pickle.loads(pickle.dumps(orig))
    self.assertEqual(pickled.default_geometry_name, orig.default_geometry_name)

  def test_pickle_fixed_lengths(self):
    orig = function.Namespace(length_i=2)
    pickled = pickle.loads(pickle.dumps(orig))
    self.assertEqual(pickled.eval_ij('δ_ij').shape, (2,2))

  def test_pickle_fallback_length(self):
    orig = function.Namespace(fallback_length=2)
    pickled = pickle.loads(pickle.dumps(orig))
    self.assertEqual(pickled.eval_ij('δ_ij').shape, (2,2))

  def test_duplicate_fixed_lengths(self):
    with self.assertRaisesRegex(ValueError, '^length of index i specified more than once$'):
      function.Namespace(length_ii=2)

  def test_unexpected_keyword_argument(self):
    with self.assertRaisesRegex(TypeError, r"^__init__\(\) got an unexpected keyword argument 'test'$"):
      function.Namespace(test=2)

  def test_d_geom(self):
    ns = function.Namespace()
    topo, ns.x = mesh.rectilinear([1])
    self.assertEqualLowered(ns.eval_ij('d(x_i, x_j)'), function.grad(ns.x, ns.x), ndims=topo.ndims)

  def test_d_arg(self):
    ns = function.Namespace()
    ns.a = '?a'
    self.assertEqual(ns.eval_('d(2 ?a + 1, ?a)').prepare_eval(npoints=None).simplified, function.asarray(2).prepare_eval(npoints=None).simplified)

  def test_n(self):
    ns = function.Namespace()
    topo, ns.x = mesh.rectilinear([1])
    self.assertEqualLowered(ns.eval_i('n(x_i)'), function.normal(ns.x), ndims=topo.ndims)

  def test_functions(self):
    def sqr(a):
      return a**2
    def mul(*args):
      if len(args) == 2:
        return args[0][(...,)+(None,)*args[1].ndim] * args[1][(None,)*args[0].ndim]
      else:
        return mul(mul(args[0], args[1]), *args[2:])
    ns = function.Namespace(functions=dict(sqr=sqr, mul=mul))
    ns.a = numpy.array([1, 2, 3])
    ns.b = numpy.array([4, 5])
    ns.A = numpy.array([[6, 7, 8], [9, 10, 11]])
    l = lambda f: f.prepare_eval(npoints=None).simplified
    self.assertEqual(l(ns.eval_i('sqr(a_i)')), l(sqr(ns.a)))
    self.assertEqual(l(ns.eval_ij('mul(a_i, b_j)')), l(ns.eval_ij('a_i b_j')))
    self.assertEqual(l(ns.eval_('mul(b_i, A_ij, a_j)')), l(ns.eval_('b_i A_ij a_j')))

  def test_builtin_functions(self):
    ns = function.Namespace()
    domain, ns.x = mesh.rectilinear([1]*2)
    ns.a = numpy.array([1, 2, 3])
    ns.A = numpy.array([[6, 7, 8], [9, 10, 11]])
    l = lambda f: f.prepare_eval(npoints=4, ndims=2).simplified
    self.assertEqual(l(ns.eval_('norm2(a)')), l(function.norm2(ns.a)))
    self.assertEqual(l(ns.eval_i('sum:j(A_ij)')), l(function.sum(ns.A, 1)))

  def test_builtin_jacobian_vector(self):
    ns = function.Namespace()
    domain, ns.x = mesh.rectilinear([1]*2)
    l = lambda f: f.prepare_eval(npoints=4, ndims=2).simplified
    self.assertEqual(l(ns.eval_('J(x)')), l(function.jacobian(ns.x)))

  def test_builtin_jacobian_scalar(self):
    ns = function.Namespace()
    domain, (ns.t,) = mesh.rectilinear([1])
    l = lambda f: f.prepare_eval(npoints=4, ndims=1).simplified
    self.assertEqual(l(ns.eval_('J(t)')), l(function.jacobian(ns.t[None])))

  def test_builtin_jacobian_matrix(self):
    ns = function.Namespace()
    ns.x = numpy.array([[1,2],[3,4]])
    with self.assertRaises(ValueError):
      ns.eval_('J(x)')

  def test_builtin_jacobian_vectorization(self):
    with self.assertRaises(NotImplementedError):
      function._J_expr(function.Array.cast([[1,2],[3,4]]), consumes=1)

class eval_ast(TestCase):

  def setUp(self):
    super().setUp()
    domain, x = mesh.rectilinear([2,2])
    self.ns = function.Namespace()
    self.ns.x = x
    self.ns.altgeom = function.concatenate([self.ns.x, [0]], 0)
    self.ns.basis = domain.basis('spline', degree=2)
    self.ns.a = 2
    self.ns.a2 = numpy.array([1,2])
    self.ns.a3 = numpy.array([1,2,3])
    self.ns.a22 = numpy.array([[1,2],[3,4]])
    self.ns.a32 = numpy.array([[1,2],[3,4],[5,6]])
    self.x = function.Argument('x',())

  def assertEqualLowered(self, s, f):
    self.assertEqual((s @ self.ns).prepare_eval(ndims=2).simplified, f.prepare_eval(ndims=2).simplified)

  def test_group(self): self.assertEqualLowered('(a)', self.ns.a)
  def test_arg(self): self.assertEqualLowered('a2_i ?x_i', function.dot(self.ns.a2, function.Argument('x', [2]), axes=[0]))
  def test_substitute(self): self.assertEqualLowered('(?x_i^2)(x_i=a2_i)', self.ns.a2**2)
  def test_multisubstitute(self): self.assertEqualLowered('(a2_i + ?x_i + ?y_i)(x_i=?y_i, y_i=?x_i)', self.ns.a2 + function.Argument('y', [2]) + function.Argument('x', [2]))
  def test_call(self): self.assertEqualLowered('sin(a)', function.sin(self.ns.a))
  def test_call2(self): self.assertEqual(self.ns.eval_ij('arctan2(a2_i, a3_j)').prepare_eval(ndims=2).simplified, function.arctan2(self.ns.a2[:,None], self.ns.a3[None,:]).prepare_eval(ndims=2).simplified)
  def test_eye(self): self.assertEqualLowered('δ_ij a2_i', function.dot(function.eye(2), self.ns.a2, axes=[0]))
  def test_normal(self): self.assertEqualLowered('n_i', self.ns.x.normal())
  def test_getitem(self): self.assertEqualLowered('a2_0', self.ns.a2[0])
  def test_trace(self): self.assertEqualLowered('a22_ii', function.trace(self.ns.a22, 0, 1))
  def test_sum(self): self.assertEqualLowered('a2_i a2_i', function.sum(self.ns.a2 * self.ns.a2, axis=0))
  def test_concatenate(self): self.assertEqualLowered('<a, a>_i', function.concatenate([self.ns.a[None],self.ns.a[None]], axis=0))
  def test_grad(self): self.assertEqualLowered('basis_n,0', self.ns.basis.grad(self.ns.x)[:,0])
  def test_surfgrad(self): self.assertEqualLowered('surfgrad(basis_0, altgeom_i)', function.grad(self.ns.basis[0], self.ns.altgeom, len(self.ns.altgeom)-1))
  def test_derivative(self): self.assertEqualLowered('d(exp(?x), ?x)', function.derivative(function.exp(self.x), self.x))
  def test_append_axis(self): self.assertEqualLowered('a a2_i', self.ns.a[None]*self.ns.a2)
  def test_transpose(self): self.assertEqualLowered('a22_ij a22_ji', function.dot(self.ns.a22, self.ns.a22.T, axes=[0,1]))
  def test_jump(self): self.assertEqualLowered('[a]', function.jump(self.ns.a))
  def test_mean(self): self.assertEqualLowered('{a}', function.mean(self.ns.a))
  def test_neg(self): self.assertEqualLowered('-a', -self.ns.a)
  def test_add(self): self.assertEqualLowered('a + ?x', self.ns.a + self.x)
  def test_sub(self): self.assertEqualLowered('a - ?x', self.ns.a - self.x)
  def test_mul(self): self.assertEqualLowered('a ?x', self.ns.a * self.x)
  def test_truediv(self): self.assertEqualLowered('a / ?x', self.ns.a / self.x)
  def test_pow(self): self.assertEqualLowered('a^2', self.ns.a**2)

  def test_unknown_opcode(self):
    with self.assertRaises(ValueError):
      function._eval_ast(('invalid-opcode',), {})

  def test_call_invalid_shape(self):
    with self.assertRaisesRegex(ValueError, '^expected an array with shape'):
      function._eval_ast(('call', (None, 'f'), (None, 0), (None, 0), (None, function.zeros((2,), float)), (None, function.zeros((3,), float))),
                         dict(f=lambda a, b: a[None,:] * b[:,None])) # result is transposed

  def test_surfgrad_deprecated(self):
    with self.assertWarns(warnings.NutilsDeprecationWarning):
      self.assertEqualLowered('basis_n;altgeom_0', function.grad(self.ns.basis, self.ns.altgeom, len(self.ns.altgeom)-1)[:,0])

  def test_derivative_deprecated(self):
    with self.assertWarns(warnings.NutilsDeprecationWarning):
      self.assertEqualLowered('exp(?x)_,?x', function.derivative(function.exp(self.x), self.x))

class jacobian(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.unitsquare(1, 'square')
    self.basis = self.domain.basis('std', degree=1)
    arg = function.Argument('dofs', [4])
    self.v = self.basis.dot(arg)
    self.X = (self.geom[numpy.newaxis,:] * [[0,1],[-self.v,0]]).sum(-1) # X_i = <x_1, -2 x_0>_i
    self.J = function.J(self.X)
    self.dJ = function.derivative(self.J, arg)

  def test_shape(self):
    self.assertEqual(self.J.shape, ())
    self.assertEqual(self.dJ.shape, (4,))

  def test_value(self):
    values = self.domain.sample('uniform', 2).eval(self.J, dofs=[2]*4)
    numpy.testing.assert_almost_equal(values, [2]*4)
    values1, values2 = self.domain.sample('uniform', 2).eval([self.J,
      self.v + self.v.grad(self.geom)[0] * self.geom[0]], dofs=[1,2,3,10])
    numpy.testing.assert_almost_equal(values1, values2)

  def test_derivative(self):
    values1, values2 = self.domain.sample('uniform', 2).eval([self.dJ,
      self.basis + self.basis.grad(self.geom)[:,0] * self.geom[0]], dofs=[1,2,3,10])
    numpy.testing.assert_almost_equal(values1, values2)

  def test_zeroderivative(self):
    otherarg = function.Argument('otherdofs', (10,))
    values = self.domain.sample('uniform', 2).eval(function.derivative(self.dJ, otherarg))
    self.assertEqual(values.shape[1:], self.dJ.shape + otherarg.shape)
    self.assertAllEqual(values, 0)

@parametrize
class derivative(TestCase):

  def assertEvalAlmostEqual(self, topo, factual, fdesired):
    actual, desired = topo.sample('uniform', 2).eval([function.asarray(factual), function.asarray(fdesired)])
    self.assertAllAlmostEqual(actual, desired)

  def test_grad_0d(self):
    domain, (x,) = mesh.rectilinear([1])
    x = 2*x-0.5
    self.assertEvalAlmostEqual(domain, self.grad(x**2, x), 2*x)

  def test_grad_1d(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    self.assertEvalAlmostEqual(domain, self.grad([x[0]**2*x[1], x[1]**2], x), [[2*x[0]*x[1], x[0]**2], [0, 2*x[1]]])

  def test_grad_2d(self):
    domain, x = mesh.rectilinear([1]*4)
    x = 2*x-0.5
    x = function.unravel(x, 0, (2, 2))
    self.assertEvalAlmostEqual(domain, self.grad(x, x), numpy.eye(4, 4).reshape(2, 2, 2, 2))

  def test_grad_3d(self):
    domain, x = mesh.rectilinear([1]*4)
    x = 2*x-0.5
    x = function.unravel(function.unravel(x, 0, (2, 2)), 0, (2, 1))
    self.assertEvalAlmostEqual(domain, self.grad(x, x), numpy.eye(4, 4).reshape(2, 1, 2, 2, 1, 2))

  def test_div(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    self.assertEvalAlmostEqual(domain, self.div([x[0]**2+x[1], x[1]**2-x[0]], x), 2*x[0]+2*x[1])

  def test_laplace(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    self.assertEvalAlmostEqual(domain, self.laplace(x[0]**2*x[1]-x[1]**2, x), 2*x[1]-2)

  def test_symgrad(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    self.assertEvalAlmostEqual(domain, self.symgrad([x[0]**2*x[1], x[1]**2], x), [[2*x[0]*x[1], 0.5*x[0]**2], [0.5*x[0]**2, 2*x[1]]])

  def test_normal_0d(self):
    domain, (x,) = mesh.rectilinear([1])
    x = 2*x-0.5
    self.assertEvalAlmostEqual(domain.boundary['right'], self.normal(x), 1)
    self.assertEvalAlmostEqual(domain.boundary['left'], self.normal(x), -1)

  def test_normal_1d(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    for bnd, n in ('right', [1, 0]), ('left', [-1, 0]), ('top', [0, 1]), ('bottom', [0, -1]):
      self.assertEvalAlmostEqual(domain.boundary[bnd], self.normal(x), n)

  def test_normal_2d(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    x = function.unravel(x, 0, [2, 1])
    for bnd, n in ('right', [1, 0]), ('left', [-1, 0]), ('top', [0, 1]), ('bottom', [0, -1]):
      self.assertEvalAlmostEqual(domain.boundary[bnd], self.normal(x), numpy.array(n)[:,_])

  def test_normal_3d(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    x = function.unravel(function.unravel(x, 0, [2, 1]), 0, [1, 2])
    for bnd, n in ('right', [1, 0]), ('left', [-1, 0]), ('top', [0, 1]), ('bottom', [0, -1]):
      self.assertEvalAlmostEqual(domain.boundary[bnd], self.normal(x), numpy.array(n)[_,:,_])

  def test_dotnorm(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    for bnd, desired in ('right', 1), ('left', -1), ('top', 0), ('bottom', 0):
      self.assertEvalAlmostEqual(domain.boundary[bnd], self.dotnorm([1, 0], x), desired)

  def test_tangent(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    for bnd, desired in ('right', [0, 1]), ('left', [0, 1]), ('top', [-1, 0]), ('bottom', [-1, 0]):
      self.assertEvalAlmostEqual(domain.boundary[bnd], self.tangent(x, [-1, 1]), desired)

  def test_ngrad(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    for bnd, desired in ('right', [2*x[0]*x[1], 0]), ('left', [-2*x[0]*x[1], 0]), ('top', [x[0]**2, 2*x[1]]), ('bottom', [-x[0]**2, -2*x[1]]):
      self.assertEvalAlmostEqual(domain.boundary[bnd], self.ngrad([x[0]**2*x[1], x[1]**2], x), desired)

  def test_nsymgrad(self):
    domain, x = mesh.rectilinear([1]*2)
    x = 2*x-0.5
    for bnd, desired in ('right', [2*x[0]*x[1], 0.5*x[0]**2]), ('left', [-2*x[0]*x[1], -0.5*x[0]**2]), ('top', [0.5*x[0]**2, 2*x[1]]), ('bottom', [-0.5*x[0]**2, -2*x[1]]):
      self.assertEvalAlmostEqual(domain.boundary[bnd], self.nsymgrad([x[0]**2*x[1], x[1]**2], x), desired)

derivative('function',
           normal=function.normal,
           tangent=function.tangent,
           dotnorm=function.dotnorm,
           grad=function.grad,
           div=function.div,
           laplace=function.laplace,
           symgrad=function.symgrad,
           ngrad=function.ngrad,
           nsymgrad=function.nsymgrad)
derivative('method',
           normal=lambda geom: function.Array.cast(geom).normal(),
           tangent=lambda geom, vec: function.Array.cast(geom).tangent(vec),
           dotnorm=lambda vec, geom: function.Array.cast(vec).dotnorm(geom),
           grad=lambda arg, geom: function.Array.cast(arg).grad(geom),
           div=lambda arg, geom: function.Array.cast(arg).div(geom),
           laplace=lambda arg, geom: function.Array.cast(arg).laplace(geom),
           symgrad=lambda arg, geom: function.Array.cast(arg).symgrad(geom),
           ngrad=lambda arg, geom: function.Array.cast(arg).ngrad(geom),
           nsymgrad=lambda arg, geom: function.Array.cast(arg).nsymgrad(geom))

class deprecations(TestCase):

  def test_simplified(self):
    with self.assertWarns(warnings.NutilsDeprecationWarning):
      function.simplified(function.Argument('a', ()))

  def test_iszero(self):
    with self.assertWarns(warnings.NutilsDeprecationWarning):
      function.iszero(function.Argument('a', ()))

class CommonBasis:

  @staticmethod
  def mk_index_coords(coorddim, transforms):
    index = function.transforms_index(transforms)
    coords = function.transforms_coords(transforms, coorddim)
    return index, coords

  def setUp(self):
    super().setUp()
    self.checknelems = len(self.checkcoeffs)
    self.checksupp = [[] for i in range(self.checkndofs)]
    for ielem, dofs in enumerate(self.checkdofs):
      for dof in dofs:
        self.checksupp[dof].append(ielem)
    assert len(self.checkcoeffs) == len(self.checkdofs)
    assert all(len(c) == len(d) for c, d in zip(self.checkcoeffs, self.checkdofs))

  def test_shape(self):
    self.assertEqual(self.basis.shape, (self.checkndofs,))

  def test_get_coefficients_pos(self):
    for ielem in range(self.checknelems):
      self.assertEqual(self.basis.get_coefficients(ielem).tolist(), self.checkcoeffs[ielem])

  def test_get_coefficients_neg(self):
    for ielem in range(-self.checknelems, 0):
      self.assertEqual(self.basis.get_coefficients(ielem).tolist(), self.checkcoeffs[ielem])

  def test_get_coefficients_outofbounds(self):
    with self.assertRaises(IndexError):
      self.basis.get_coefficients(-self.checknelems-1)
    with self.assertRaises(IndexError):
      self.basis.get_coefficients(self.checknelems)

  def test_get_dofs_scalar_pos(self):
    for ielem in range(self.checknelems):
      self.assertEqual(self.basis.get_dofs(ielem).tolist(), self.checkdofs[ielem])

  def test_get_dofs_scalar_neg(self):
    for ielem in range(-self.checknelems, 0):
      self.assertEqual(self.basis.get_dofs(ielem).tolist(), self.checkdofs[ielem])

  def test_get_dofs_scalar_outofbounds(self):
    with self.assertRaises(IndexError):
      self.basis.get_dofs(-self.checknelems-1)
    with self.assertRaises(IndexError):
      self.basis.get_dofs(self.checknelems)

  def test_get_ndofs(self):
    for ielem in range(self.checknelems):
      self.assertEqual(self.basis.get_ndofs(ielem), len(self.checkdofs[ielem]))

  def test_dofs_array(self):
    for mask in itertools.product(*[[False, True]]*self.checknelems):
      mask = numpy.array(mask, dtype=bool)
      indices, = numpy.where(mask)
      for value in mask, indices:
        with self.subTest(tuple(value)):
          self.assertEqual(self.basis.get_dofs(value).tolist(), list(sorted(set(itertools.chain.from_iterable(self.checkdofs[i] for i in indices)))))

  def test_dofs_intarray_outofbounds(self):
    for i in [-1, self.checknelems]:
      with self.assertRaises(IndexError):
        self.basis.get_dofs(numpy.array([i], dtype=int))

  def test_dofs_intarray_invalidndim(self):
    with self.assertRaises(IndexError):
      self.basis.get_dofs(numpy.array([[0]], dtype=int))

  def test_dofs_boolarray_invalidshape(self):
    with self.assertRaises(IndexError):
      self.basis.get_dofs(numpy.array([True]*(self.checknelems+1), dtype=bool))
    with self.assertRaises(IndexError):
      self.basis.get_dofs(numpy.array([[True]*self.checknelems], dtype=bool))

  def test_get_support_scalar_pos(self):
    for dof in range(self.checkndofs):
      self.assertEqual(self.basis.get_support(dof).tolist(), self.checksupp[dof])

  def test_get_support_scalar_neg(self):
    for dof in range(-self.checkndofs, 0):
      self.assertEqual(self.basis.get_support(dof).tolist(), self.checksupp[dof])

  def test_get_support_scalar_outofbounds(self):
    with self.assertRaises(IndexError):
      self.basis.get_support(-self.checkndofs-1)
    with self.assertRaises(IndexError):
      self.basis.get_support(self.checkndofs)

  def test_get_support_array(self):
    for mask in itertools.product(*[[False, True]]*self.checkndofs):
      mask = numpy.array(mask, dtype=bool)
      indices, = numpy.where(mask)
      for value in mask, indices:
        with self.subTest(tuple(value)):
          self.assertEqual(self.basis.get_support(value).tolist(), list(sorted(set(itertools.chain.from_iterable(self.checksupp[i] for i in indices)))))

  def test_get_support_intarray_outofbounds(self):
    for i in [-1, self.checkndofs]:
      with self.assertRaises(IndexError):
        self.basis.get_support(numpy.array([i], dtype=int))

  def test_get_support_intarray_invalidndim(self):
    with self.assertRaises(IndexError):
      self.basis.get_support(numpy.array([[0]], dtype=int))

  def test_get_support_boolarray(self):
    for mask in itertools.product(*[[False, True]]*self.checkndofs):
      mask = numpy.array(mask, dtype=bool)
      indices, = numpy.where(mask)
      with self.subTest(tuple(indices)):
        self.assertEqual(self.basis.get_support(mask).tolist(), list(sorted(set(itertools.chain.from_iterable(self.checksupp[i] for i in indices)))))

  def test_get_support_boolarray_invalidshape(self):
    with self.assertRaises(IndexError):
      self.basis.get_support(numpy.array([True]*(self.checkndofs+1), dtype=bool))
    with self.assertRaises(IndexError):
      self.basis.get_support(numpy.array([[True]*self.checkndofs], dtype=bool))

  def test_getitem_array(self):
    for mask in itertools.product(*[[False, True]]*self.checkndofs):
      mask = numpy.array(mask, dtype=bool)
      indices, = numpy.where(mask)
      for value in mask, indices:
        with self.subTest(tuple(value)):
          maskedbasis = self.basis[value]
          self.assertIsInstance(maskedbasis, function.Basis)
          for ielem in range(self.checknelems):
            m = numpy.asarray(numeric.sorted_contains(indices, self.checkdofs[ielem]))
            self.assertEqual(maskedbasis.get_dofs(ielem).tolist(), numeric.sorted_index(indices, numpy.compress(m, self.checkdofs[ielem], axis=0)).tolist())
            self.assertEqual(maskedbasis.get_coefficients(ielem).tolist(), numpy.compress(m, self.checkcoeffs[ielem], axis=0).tolist())

  def test_getitem_slice(self):
    maskedbasis = self.basis[1:-1]
    indices = numpy.arange(self.checkndofs)[1:-1]
    for ielem in range(self.checknelems):
      m = numpy.asarray(numeric.sorted_contains(indices, self.checkdofs[ielem]))
      self.assertEqual(maskedbasis.get_dofs(ielem).tolist(), numeric.sorted_index(indices, numpy.compress(m, self.checkdofs[ielem], axis=0)).tolist())
      self.assertEqual(maskedbasis.get_coefficients(ielem).tolist(), numpy.compress(m, self.checkcoeffs[ielem], axis=0).tolist())

  def test_getitem_slice_all(self):
    maskedbasis = self.basis[:]
    for ielem in range(self.checknelems):
      self.assertEqual(maskedbasis.get_dofs(ielem).tolist(), self.checkdofs[ielem])
      self.assertEqual(maskedbasis.get_coefficients(ielem).tolist(), self.checkcoeffs[ielem])

  def checkeval(self, ielem, points):
    result = numpy.zeros((points.npoints, self.checkndofs,), dtype=float)
    numpy.add.at(result, (slice(None),numpy.array(self.checkdofs[ielem], dtype=int)), numeric.poly_eval(numpy.array(self.checkcoeffs[ielem], dtype=float), points.coords))
    return result.tolist()

  def test_lower(self):
    ref = element.PointReference() if self.basis.coords.shape[0] == 0 else element.LineReference()**self.basis.coords.shape[0]
    points = ref.getpoints('bezier', 4)
    lowered = self.basis.prepare_eval(ndims=points.ndims)
    with _builtin_warnings.catch_warnings():
      _builtin_warnings.simplefilter('ignore', category=evaluable.ExpensiveEvaluationWarning)
      for ielem in range(self.checknelems):
        value = lowered.eval(_transforms=(self.checktransforms[ielem],), _points=points)
        if value.shape[0] == 1:
          value = numpy.tile(value, (points.npoints, 1))
        self.assertEqual(value.tolist(), self.checkeval(ielem, points))

class PlainBasis(CommonBasis, TestCase):

  def setUp(self):
    self.checktransforms = transformseq.PlainTransforms([(transform.Identifier(0,k),) for k in 'abcd'], 0)
    index, coords = self.mk_index_coords(0, self.checktransforms)
    self.checkcoeffs = [[1.],[2.,3.],[4.,5.],[6.]]
    self.checkdofs = [[0],[2,3],[1,3],[2]]
    self.basis = function.PlainBasis(self.checkcoeffs, self.checkdofs, 4, index, coords)
    self.checkndofs = 4
    super().setUp()

class DiscontBasis(CommonBasis, TestCase):

  def setUp(self):
    self.checktransforms = transformseq.PlainTransforms([(transform.Identifier(0,k),) for k in 'abcd'], 0)
    index, coords = self.mk_index_coords(0, self.checktransforms)
    self.checkcoeffs = [[1.],[2.,3.],[4.,5.],[6.]]
    self.basis = function.DiscontBasis(self.checkcoeffs, index, coords)
    self.checkdofs = [[0],[1,2],[3,4],[5]]
    self.checkndofs = 6
    super().setUp()

class MaskedBasis(CommonBasis, TestCase):

  def setUp(self):
    self.checktransforms = transformseq.PlainTransforms([(transform.Identifier(0,k),) for k in 'abcd'], 0)
    index, coords = self.mk_index_coords(0, self.checktransforms)
    parent = function.PlainBasis([[1.],[2.,3.],[4.,5.],[6.]], [[0],[2,3],[1,3],[2]], 4, index, coords)
    self.basis = function.MaskedBasis(parent, [0,2])
    self.checkcoeffs = [[1.],[2.],[],[6.]]
    self.checkdofs = [[0],[1],[],[1]]
    self.checkndofs = 2
    super().setUp()

class PrunedBasis(CommonBasis, TestCase):

  def setUp(self):
    parent_transforms = transformseq.PlainTransforms([(transform.Identifier(0,k),) for k in 'abcd'], 0)
    parent_index, parent_coords = self.mk_index_coords(0, parent_transforms)
    indices = types.frozenarray([0,2])
    self.checktransforms = parent_transforms[indices]
    index, coords = self.mk_index_coords(0, self.checktransforms)
    parent = function.PlainBasis([[1.],[2.,3.],[4.,5.],[6.]], [[0],[2,3],[1,3],[2]], 4, parent_index, parent_coords)
    self.basis = function.PrunedBasis(parent, indices, index, coords)
    self.checkcoeffs = [[1.],[4.,5.]]
    self.checkdofs = [[0],[1,2]]
    self.checkndofs = 3
    super().setUp()

class StructuredBasis1D(CommonBasis, TestCase):

  def setUp(self):
    self.checktransforms = transformseq.StructuredTransforms(transform.Identifier(1, 'test'), [transformseq.DimAxis(0,4,0,False)], 0)
    index, coords = self.mk_index_coords(1, self.checktransforms)
    self.basis = function.StructuredBasis([[[[1],[2]],[[3],[4]],[[5],[6]],[[7],[8]]]], [[0,1,2,3]], [[2,3,4,5]], [5], [4], index, coords)
    self.checkcoeffs = [[[1.],[2.]],[[3.],[4.]],[[5.],[6.]],[[7.],[8.]]]
    self.checkdofs = [[0,1],[1,2],[2,3],[3,4]]
    self.checkndofs = 5
    super().setUp()

class StructuredBasis1DPeriodic(CommonBasis, TestCase):

  def setUp(self):
    self.checktransforms = transformseq.StructuredTransforms(transform.Identifier(1, 'test'), [transformseq.DimAxis(0,4,4,True)], 0)
    index, coords = self.mk_index_coords(1, self.checktransforms)
    self.basis = function.StructuredBasis([[[[1],[2]],[[3],[4]],[[5],[6]],[[7],[8]]]], [[0,1,2,3]], [[2,3,4,5]], [4], [4], index, coords)
    self.checkcoeffs = [[[1.],[2.]],[[3.],[4.]],[[5.],[6.]],[[7.],[8.]]]
    self.checkdofs = [[0,1],[1,2],[2,3],[3,0]]
    self.checkndofs = 4
    super().setUp()

class StructuredBasis2D(CommonBasis, TestCase):

  def setUp(self):
    self.checktransforms = transformseq.StructuredTransforms(transform.Identifier(2, 'test'), [transformseq.DimAxis(0,2,0,False),transformseq.DimAxis(0,2,0,False)], 0)
    index, coords = self.mk_index_coords(2, self.checktransforms)
    self.basis = function.StructuredBasis([[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]], [[0,1],[0,1]], [[2,3],[2,3]], [3,3], [2,2], index, coords)
    self.checkcoeffs = [[[[5.]],[[6.]],[[10.]],[[12.]]],[[[7.]],[[8.]],[[14.]],[[16.]]],[[[15.]],[[18.]],[[20.]],[[24.]]],[[[21.]],[[24.]],[[28.]],[[32.]]]]
    self.checkdofs = [[0,1,3,4],[1,2,4,5],[3,4,6,7],[4,5,7,8]]
    self.checkndofs = 9
    super().setUp()
