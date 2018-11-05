import importlib

# Handle import of classes in fluxion_node differently depending on whether this
# module is being loaded as __main__ or a module in a package.
if importlib.util.find_spec("fluxions") is not None:
    from fluxions import Fluxion, Unop
    from fluxions import sin, cos, tan
else:
    from fluxion_node import Fluxion, Unop
    from elementary_functions import sin, cos, tan


class Sin(Unop):
    """Sin(x) as a node on an calculation graph"""
    def __init__(self, x: Fluxion):
        self.x = x
    
    def val(self):
        return sin.val(self.x.val())
    
    def diff(self):
        return sin.diff(self.x.val()) * self.x.diff()
    
class Cos(Unop):
    """Cos(x) as a node on an calculation graph"""
    def __init__(self, x: Fluxion):
        self.x = x
    
    def val(self):
        return cos.val(self.x.val())
    
    def diff(self):
        return cos.diff(self.x.val()) * self.x.diff()

class Tan(Unop):
    """Tan(x) as a node on an calculation graph"""
    def __init__(self, x: Fluxion):
        self.x = x
    
    def val(self):
        return tan.val(self.x.val())
    
    def diff(self):
        return tan.diff(self.x.val()) * self.x.diff()
