
# copied from pypy
def apply(function, args=(), kwds={}):
    """call a function (or other callable object) and return its result"""
    return function(*args, **kwds)

# ____________________________________________________________

def sorted(lst, cmp=None, key=None, reverse=False):
    "sorted(iterable, cmp=None, key=None, reverse=False) --> new sorted list"
    sorted_lst = list(lst)
    sorted_lst.sort(cmp, key, reverse)
    return sorted_lst

def any(seq):
    """any(iterable) -> bool

Return True if bool(x) is True for any x in the iterable."""
    for x in seq:
        if x:
            return True
    return False

def all(seq):
    """all(iterable) -> bool

Return True if bool(x) is True for all values x in the iterable."""
    for x in seq:
        if not x:
            return False
    return True

def sum(sequence, start=0):
    """sum(sequence[, start]) -> value

Returns the sum of a sequence of numbers (NOT strings) plus the value
of parameter 'start' (which defaults to 0).  When the sequence is
empty, returns start."""
    if isinstance(start, basestring):
        raise TypeError("sum() can't sum strings")
    last = start
    for x in sequence:
        # Very intentionally *not* +=, that would have different semantics if
        # start was a mutable type, such as a list
        last = last + x
    return last

class _Cons(object):
    def __init__(self, prev, iter):
        self.prev = prev
        self.iter = iter

    def fetch(self):
        # recursive, loop-less version of the algorithm: works best for a
        # fixed number of "collections" in the call to map(func, *collections)
        prev = self.prev
        if prev is None:
            args1 = ()
            stop = True
        else:
            args1, stop = prev.fetch()
        iter = self.iter
        if iter is None:
            val = None
        else:
            try:
                val = next(iter)
                stop = False
            except StopIteration:
                self.iter = None
                val = None
        return args1 + (val,), stop

def map(func, *collections):
    """map(function, sequence[, sequence, ...]) -> list

Return a list of the results of applying the function to the items of
the argument sequence(s).  If more than one sequence is given, the
function is called with an argument list consisting of the corresponding
item of each sequence, substituting None for missing values when not all
sequences have the same length.  If the function is None, return a list of
the items of the sequence (or a list of tuples if more than one sequence)."""
    if not collections:
        raise TypeError("map() requires at least two arguments")
    num_collections = len(collections)
    none_func = func is None
    if num_collections == 1:
        if none_func:
            return list(collections[0])
        # Special case for the really common case of a single collection
        seq = collections[0]
        with _ManagedNewlistHint(operator._length_hint(seq, 0)) as result:
            for item in seq:
                result.append(func(item))
            return result

    # Gather the iterators into _Cons objects and guess the
    # result length (the max of the input lengths)
    c = None
    max_hint = 0
    for seq in collections:
        c = _Cons(c, iter(seq))
        max_hint = max(max_hint, operator._length_hint(seq, 0))

    with _ManagedNewlistHint(max_hint) as result:
        while True:
            args, stop = c.fetch()
            if stop:
                return result
            if none_func:
                result.append(args)
            else:
                result.append(func(*args))

class _ManagedNewlistHint(object):
    """ Context manager returning a newlist_hint upon entry.

    Upon exit the list's underlying capacity will be cut back to match
    its length if necessary (incase the initial length_hint was too
    large).
    """

    def __init__(self, length_hint):
        self.length_hint = length_hint
        self.list = newlist_hint(length_hint)

    def __enter__(self):
        return self.list

    def __exit__(self, type, value, tb):
        if type is None:
            extended = len(self.list)
            if extended < self.length_hint:
                resizelist_hint(self.list, extended)

sentinel = object()

def reduce(func, sequence, initial=sentinel):
    """reduce(function, sequence[, initial]) -> value

Apply a function of two arguments cumulatively to the items of a sequence,
from left to right, so as to reduce the sequence to a single value.
For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
of the sequence in the calculation, and serves as a default when the
sequence is empty."""
    iterator = iter(sequence)
    if initial is sentinel:
        try:
            initial = next(iterator)
        except StopIteration:
            raise TypeError("reduce() of empty sequence with no initial value")
    result = initial
    for item in iterator:
        result = func(result, item)
    return result

def filter(func, seq):
    """filter(function or None, sequence) -> list, tuple, or string

Return those items of sequence for which function(item) is true.  If
function is None, return the items that are true.  If sequence is a tuple
or string, return the same type, else return a list."""
    if func is None:
        func = bool
    if isinstance(seq, str):
        return _filter_string(func, seq, str)
    elif isinstance(seq, unicode):
        return _filter_string(func, seq, unicode)
    elif isinstance(seq, tuple):
        return _filter_tuple(func, seq)
    with _ManagedNewlistHint(operator._length_hint(seq, 0)) as result:
        for item in seq:
            if func(item):
                result.append(item)
    return result

def _filter_string(func, string, str_type):
    if func is bool and type(string) is str_type:
        return string
    length = len(string)
    result = newlist_hint(length)
    for i in range(length):
        # You must call __getitem__ on the strings, simply iterating doesn't
        # work :/
        item = string[i]
        if func(item):
            if not isinstance(item, str_type):
                raise TypeError("__getitem__ returned a non-string type")
            result.append(item)
    return str_type().join(result)

def _filter_tuple(func, seq):
    length = len(seq)
    result = newlist_hint(length)
    for i in range(length):
        # Again, must call __getitem__, at least there are tests.
        item = seq[i]
        if func(item):
            result.append(item)
    return tuple(result)

def zip(*sequences):
    """zip(seq1 [, seq2 [...]]) -> [(seq1[0], seq2[0] ...), (...)]

Return a list of tuples, where each tuple contains the i-th element
from each of the argument sequences.  The returned list is truncated
in length to the length of the shortest argument sequence."""
    l = len(sequences)
    if l == 2:
        # A very fast path if the two sequences are lists
        seq0 = sequences[0]
        seq1 = sequences[1]
        try:
            return specialized_zip_2_lists(seq0, seq1)
        except TypeError:
            pass
        # This is functionally the same as the code below, but more
        # efficient because it unrolls the loops over 'sequences'.
        # Only for two arguments, which is the most common case.
        iter0 = iter(seq0)
        iter1 = iter(seq1)
        hint = min(100000000,   # max 100M
                   operator._length_hint(seq0, 0),
                   operator._length_hint(seq1, 0))

        with _ManagedNewlistHint(hint) as result:
            while True:
                try:
                    item0 = next(iter0)
                    item1 = next(iter1)
                except StopIteration:
                    return result
                result.append((item0, item1))

    if l == 0:
        return []

    # Gather the iterators and guess the result length (the min of the
    # input lengths).  If any of the iterators doesn't know its length,
    # we use 0 (instead of ignoring it and using the other iterators;
    # see lib-python's test_builtin.test_zip).
    iterators = []
    hint = 100000000   # max 100M
    for seq in sequences:
        iterators.append(iter(seq))
        hint = min(hint, operator._length_hint(seq, 0))

    with _ManagedNewlistHint(hint) as result:
        while True:
            try:
                items = [next(it) for it in iterators]
            except StopIteration:
                return result
            result.append(tuple(items))


def enumerate(iterable):
    idx = 0
    for item in iterable:
        yield((idx, item))
        idx += 1


def abs(number):
    if hasattr(number, '__abs__'):
        return number.__abs__()

    if number < 0:
        return number * -1
    else:
        return number


class range:
    def __init__(self, start, stop, step=1):
        if not stop:
            stop = start
            start = 0

        if stop < start and step > 0 or \
           stop > start and step < 0:
            raise IndexError

        self.start = start
        self.stop = stop
        self.step = step

        self.iterstop = abs((stop - start) / step)
        self.cvalue = self.start
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.iterstop:
            self.cvalue += self.step
            self.current += 1
            return self.cvalue - self.step
        else:
            raise StopIteration

xrange = range

def _minmax(cmpfct, iterable, *args, key=None, default=None):
    if len(args) > 0:
        iterable = [iterable] + [i for i in args]
    curr_max = None
    if len(iterable) == 0:
        if default:
            return default
        else:
            raise ValueError('minmax() arg is an empty sequence')
    for item in iterable:
        if key:
            cmp_item = item[key]
        else:
            cmp_item = item
        if curr_max and cmpfct(curr_max, cmp_item):
            curr_max = cmp_item
        if not curr_max:
            curr_max = cmp_item
    return curr_max

def max(iterable, *args, key=None, default=None):
    return _minmax(lambda x, y: x < y, iterable, *args, key=key, default=default)

def min(iterable, *args, key=None, default=None):
    return _minmax(lambda x, y: x > y, iterable, *args, key=key, default=default)


# Missing functions
# chr, ord --> ?

def divmod(x, y):
    # TODO implement for float values!
    return (x // y, x % y)

def bool(x):
    # adapted from brython JS
    if not x:
        return False
    if isinstance(x, bool):
        return x
    elif isinstance(x, int) or isinstance(x, float) or isinstance(x, basestring):
        if x:
            return True
        else:
            return False
    else:
        try:
            return getattr(x, '__bool__')()
        except:
            try:
                return getattr(x, '__len__')() > 0
            except:
                return True


def _ntostring(number, base, prefix):
    if not isinstance(number, int):
        try:
            number = number.__index__()
        except:
            raise TypeError('%s object cannot be interpreted as integer' % (number.__class__.__name__))

    if number < 0:
        prefix = '-' + prefix
    number = abs(number)
    x = 0
    while base ** x < number:
        x += 1
    x = x - 1 if x > 0 else 0
    res = ''
    intmapping = '0123456789abcdefghijklmnopqrstuvwxyz'
    while x >= 0:
        t, rest = divmod(number, (base ** (x)))
        res += intmapping[t]
        number = rest
        x -= 1
        # print(t, number, rest)

    return prefix + res

def hex(number):
    return _ntostring(number, base=16, prefix='0x')

def bin(number):
    return _ntostring(number, base=2, prefix='0b')

def oct(number):
    return _ntostring(number, base=8, prefix='0o')

class memoryview:
    def __init__(self, obj):
        # TODO check that value supports BufferInterface
        self.obj = [i for i in obj]
        self.readonly = False

    def __getitem__(self, index):
        return self.obj[index]

    def __setitem__(self, rng, value):
        # TODO check that value supports BufferInterface
        olen = len(self.obj)
        slice_sizes = rng.indices(olen)
        change_len = (slice_sizes[1] - slice_sizes[0]) / slice_sizes[2]
        if change_len < len(value):
            raise IndexError
        self.obj[rng] = value

    def tobytes(self):
        return bytes(self.obj)

    def tolist(self):
        return self.obj

    def release(self):
        self.obj = None


class complex:
    def __init__(self, real, imag=None):
        if isinstance(real, str):
            self._parse(real)
        else:
            self.real = real
            self.imag = imag

    def __add__(self, other):
        if isinstance(other, complex):
            nr = self.real + other.real
            ni = self.imag + other.imag
            return complex(nr, ni)

    def __repr__(self):
        return "(%r+%rj)" % (self.real, self.imag)

def repr(object):
    return object.__repr__()

def test():
    l = ['a', 'b', 'c']
    for i in enumerate(l):
        print(i)

    a = memoryview(b'123')
    a2 = __builtins__.memoryview(bytearray(b'123'))
    print(a[1:3])
    a[1:3] = b'12'
    a2[1:3] = b'12'
    print(a[:])

    print(a)

    for i in range(0, 5):
        print(i)
    for i in range(0, 5, 10):
        print(i)

    for i in range(-1, -5, -1):
        print(i)
    for i in xrange(0, -5, -1):
        print(i)

    print(42, bin(42), __builtins__.bin(42))
    print(42, hex(42), __builtins__.hex(42))
    print(21341, hex(21341), __builtins__.hex(21341))
    print(-21341, hex(-21341), __builtins__.hex(-21341))
    print(-21341, oct(-21341), __builtins__.oct(-21341))
    print(-0, oct(-0), __builtins__.oct(-0))
    print(-0, bin(-0), __builtins__.bin(-0))
    try:
        print(-0.2, bin(-0.2), __builtins__.bin(-0.2))
    except:
        pass
    # print(-1042, bin(-1042))

    print(max([1,2,3,4,-1,-2312, 10023021]))
    print(min([1,2,3,4,-1,-2312, 10023021]))
    print(min([], default=-1000))
    try:
        print(min([]))
    except ValueError as e:
        print("correct value error", e)

    print(a.tobytes())
    print(a2.tobytes())
    print(a.tolist())
    print(a2.tolist())

    c = complex(3.2, 5)
    c2 = complex(4, 6)
    print(c + c2)
    print(__builtins__.complex(3.2,5) + __builtins__.complex(4,6))


if __name__ == '__main__':
    test()