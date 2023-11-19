def commutative_addition(cls):
    def __radd__(self, other):
        return self.__add__(other)

    cls.__radd__ = __radd__
    return cls


def commutative_multiplication(cls):
    def __rmul__(self, other):
        return self.__mul__(other)

    cls.__rmul__ = __rmul__
    return cls


def subtract_using_negative(cls):
    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    cls.__sub__ = __sub__
    cls.__rsub__ = __rsub__
    return cls
