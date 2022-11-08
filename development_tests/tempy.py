class test1():
    def __init__(self, bound):
            if bound:
                    def adjust(x):
                            return 1 - x
            else:
                    def adjust(x):
                            return x
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self, x):
            #  return adjust(x)
            return 1-x
