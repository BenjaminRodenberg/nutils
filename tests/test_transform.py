from nutils import *
from nutils.testing import *

class specialcases(TestCase):

  def test_tensoredge_swapup_identifier(self):
    lineedge = transform.SimplexEdge(1, 0, False)
    for edge in transform.TensorEdge1(lineedge, 1), transform.TensorEdge2(1, lineedge):
      with self.subTest(type(edge).__name__):
        idnt = transform.Identifier(1, 'test')
        self.assertEqual(edge.swapup(idnt), None)

class TestTransform(TestCase):

  def setUp(self, trans, linear, offset):
    self.trans = trans
    self.linear = linear
    self.offset = offset

  def test_fromdim(self):
    self.assertEqual(self.trans.fromdim, numpy.shape(self.linear)[1])

  def test_todim(self):
    self.assertEqual(self.trans.todim, numpy.shape(self.linear)[0])

  def test_fromdims(self):
    with self.assertWarns(warnings.NutilsDeprecationWarning):
      self.assertEqual(self.trans.fromdims, numpy.shape(self.linear)[1])

  def test_todims(self):
    with self.assertWarns(warnings.NutilsDeprecationWarning):
      self.assertEqual(self.trans.todims, numpy.shape(self.linear)[0])

  def test_linear(self):
    self.assertAllEqual(self.trans.linear, self.linear)

  def test_offset(self):
    self.assertAllEqual(self.trans.offset, self.offset)

  def test_apply(self):
    coords = numpy.array([[0]*self.trans.fromdim, numpy.arange(.5,self.trans.fromdim)/self.trans.fromdim])
    a, b = self.trans.apply(coords)
    self.assertAllAlmostEqual(a, self.offset)
    self.assertAllAlmostEqual(b, numpy.dot(self.linear, coords[1]) + self.offset)

class TestInvertible(TestTransform):

  def test_invapply(self):
    coords = numpy.array([self.offset, numpy.arange(.5,self.trans.fromdim)/self.trans.fromdim])
    a, b = self.trans.invapply(coords)
    self.assertAllAlmostEqual(a, 0)
    self.assertAllAlmostEqual(b, numpy.linalg.solve(self.linear, (coords[1] - self.offset)))

class TestUpdim(TestTransform):

  def test_ext(self):
    ext = numeric.ext(self.linear)
    self.assertAllAlmostEqual(ext, self.trans.ext)

class Matrix(TestTransform):

  def setUp(self):
    super().setUp(trans=transform.Matrix([[1.],[2]], [3.,4]), linear=[[1],[2]], offset=[3,4])

class Qquare(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.Square([[1.,2],[1,3]], [5.,6]), linear=[[1,2],[1,3]], offset=[5,6])

class Shift(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.Shift([1.,2]), linear=[[1,0],[0,1]], offset=[1,2])

class Identity(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.Identity(2), linear=[[1,0],[0,1]], offset=[0,0])

class Scale(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.Scale(2, offset=[1.,2]), linear=[[2,0],[0,2]], offset=[1,2])

class SimplexEdge(TestUpdim):

  def setUp(self):
    super().setUp(trans=transform.SimplexEdge(3, 0), linear=[[-1.,-1],[1,0],[0,1]], offset=[1,0,0])

class SimplexChild(TestInvertible):

  def setUp(self):
    super().setUp(trans=transform.SimplexChild(3, 1), linear=numpy.eye(3)/2, offset=[.5,0,0])

del TestTransform, TestInvertible, TestUpdim


class TestEvaluableTransformChain:

  # requires attributes `echain`, `chain` and `evalargs`

  def test_evalf(self):
    self.assertEqual(self.echain.eval(**self.evalargs), self.chain)

  def test_todim(self):
    self.assertEqual(self.echain.todim.eval(**self.evalargs), self.chain.todim)

  def test_fromdim(self):
    self.assertEqual(self.echain.fromdim.eval(**self.evalargs), self.chain.fromdim)

  def test_linear(self):
    self.assertAllAlmostEqual(self.echain.linear.eval(**self.evalargs), self.chain.linear)

  def test_linear_derivative(self):
    self.assertTrue(evaluable.iszero(evaluable.derivative(self.echain.linear, evaluable.Argument('test', ())).simplified))

  def test_extended_linear(self):
    self.assertAllAlmostEqual(self.echain.extended_linear.eval(**self.evalargs), self.chain.extended_linear)

  def test_extended_linear_derivative(self):
    self.assertTrue(evaluable.iszero(evaluable.derivative(self.echain.extended_linear, evaluable.Argument('test', ())).simplified))

  def test_apply(self):
    todim, fromdim = evaluable.Tuple((self.echain.todim, self.echain.fromdim)).eval(**self.evalargs)
    epoints = evaluable.Argument('points', (5, self.echain.fromdim), float)
    points = numpy.linspace(0, 1, 5*fromdim).reshape(5, fromdim)
    self.assertAllAlmostEqual(self.echain.apply(epoints).eval(points=points, **self.evalargs), self.chain.apply(points))

  def test_apply_derivative(self):
    todim, fromdim = evaluable.Tuple((self.echain.todim, self.echain.fromdim)).eval(**self.evalargs)
    epoints = evaluable.Argument('points', (5, self.echain.fromdim), float)
    points = numpy.linspace(0, 1, 5*fromdim).reshape(5, fromdim)
    actual = evaluable.derivative(self.echain.apply(epoints), epoints).eval(**self.evalargs)
    desired = numpy.einsum('jk,iklm->ijlm', self.chain.linear, numpy.eye(5*fromdim).reshape(5, fromdim, 5, fromdim))
    self.assertAllAlmostEqual(actual, desired)

class TestEvaluableTransformChains(TestEvaluableTransformChain):

  def test_nchains(self):
    self.assertEqual(self.echain.nchains, self.chain.nchains)

  def test_len(self):
    self.assertEqual(len(self.echain), self.chain.nchains)

  def test_get_chain(self):
    for i in range(len(self.chain)):
      self.assertEqual(self.echain.get_chain(i).eval(**self.evalargs), self.chain.get_chain(i))

  def test_getitem(self):
    for i in range(len(self.chain)):
      self.assertEqual(self.echain[i].eval(**self.evalargs), self.chain.get_chain(i))

  def test_iter(self):
    echains = tuple(iter(self.echain))
    self.assertEqual(len(echains), self.chain.nchains)
    for echain, chain in zip(echains, self.chain):
      self.assertEqual(echain.eval(**self.evalargs), chain)

  def test_reversed(self):
    echains = tuple(reversed(self.echain))
    self.assertEqual(len(echains), self.chain.nchains)
    for echain, chain in zip(echains, reversed(self.chain)):
      self.assertEqual(echain.eval(**self.evalargs), chain)

  def todims(self):
    etodims = self.echain.todims
    self.assertEqual(len(etodims), self.chain.nchains)
    for etodim, todim in zip(etodims, self.chain.todims):
      self.assertEqual(etodim.eval(**self.evalargs), todim)

  def fromdims(self):
    efromdims = self.echain.fromdims
    self.assertEqual(len(efromdims), len(self.chain))
    for efromdim, fromdim in zip(efromdims, self.chain.fromdims):
      self.assertEqual(efromdim.eval(**self.evalargs), fromdim)

class TestEvaluableTransformChainArgumentWithoutDim(TestCase, TestEvaluableTransformChain):

  def setUp(self):
    self.echain = transform.EvaluableTransformChain.from_argument('chain')
    self.chain = transform.TransformChain(transform.SimplexEdge(2, 0))
    self.evalargs = dict(chain=self.chain)

class TestEvaluableTransformChainArgumentWithDim(TestCase, TestEvaluableTransformChain):

  def setUp(self):
    self.echain = transform.EvaluableTransformChain.from_argument('chain', todim=2, fromdim=1)
    self.chain = transform.TransformChain(transform.SimplexEdge(2, 0))
    self.evalargs = dict(chain=self.chain)

class TestEvaluableTransformChainsArgumentWithoutDim(TestCase, TestEvaluableTransformChains):

  def setUp(self):
    self.echain = transform.EvaluableTransformChains.from_argument('chain', (None, None))
    self.chain = transform.TransformChains(transform.TransformChain(transform.Identity(2)), transform.TransformChain(transform.SimplexEdge(2, 0)))
    self.evalargs = dict(chain=self.chain)

class TestEvaluableTransformChainsArgumentWithDim(TestCase, TestEvaluableTransformChains):

  def setUp(self):
    self.echain = transform.EvaluableTransformChains.from_argument('chain', (2, 2), (2, 1))
    self.chain = transform.TransformChains(transform.TransformChain(transform.Identity(2)), transform.TransformChain(transform.SimplexEdge(2, 0)))
    self.evalargs = dict(chain=self.chain)

class TestEvaluableTransformChainsJoin(TestCase, TestEvaluableTransformChains):

  def setUp(self):
    echain1 = transform.EvaluableTransformChain.from_argument('chain1', todim=2, fromdim=2)
    echain2 = transform.EvaluableTransformChain.from_argument('chain2', todim=2, fromdim=1)
    chain1 = transform.TransformChain(transform.Identity(2))
    chain2 = transform.TransformChain(transform.SimplexEdge(2, 0))
    self.echain = transform.EvaluableTransformChains.from_individual_chains(echain1, echain2)
    self.chain = transform.TransformChains(chain1, chain2)
    self.evalargs = dict(chain1=chain1, chain2=chain2)
