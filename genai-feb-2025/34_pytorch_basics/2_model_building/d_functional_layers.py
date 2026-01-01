class Dummy1:
    def __init__(self):
        pass

    def forward(self, x):
        print(f"Forward method called with {x}")
        return x * 2


class Dummy2:
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        print(f"Forward method called with {x}")
        return x * 2


d1 = Dummy1()
print(d1.forward(5))
# print(d1(5))

d2 = Dummy2()
print(d2.forward(5))
print(d2(5))
