# Author: Wolf Vollprecht <w.vollprecht@gmail.com>.

# This is a stump implementation implementing the array 
# interface but giving you none of the speed boosts 
# a C implementation would give you

# TODO add type checks

class array(list):

	typecodes = 'bBuhHiIlLqQfd'
	def __init__(self, typecode, initializer=None):
		self.typecode = typecode
		# TODO
		self.itemsize = 4

		if initializer:
			if isinstance(initializer, list):
				self.fromlist(initializer)
			elif isinstance(initializer, bytes):
				self.frombytes(initializer)
			elif isinstance(initializer, str):
				self.fromunicode(initializer)
			else:
				self.extend(initializer)


	def byteswap(self):
		pass

	def fromanything(self, s):
		self.extend([i for i in s])

	def frombytes(self, s):
		self.fromanything(s)
	fromstring = frombytes

	def fromlist(self, s):
		self.fromanything(s)

	def fromunicode(self, s):
		if not isinstance(s, str) or self.typecode not in 'u':
			raise TypeError
		self.fromanything(s)

	def tobytes(self):
		return bytes(''.join(self))
	tostring = tobytes

	def tolist(self):
		return list(self)

	def tounicode(self):
		return str(''.join(self))

	def __repr__(self):
		return 'array(%r, %r)' % (self.typecode, self.tounicode())