from ltl.grammar import AtomicProposition

class SafeFromLava(AtomicProposition):
    def __init__(self, id) -> None:
        super().__init__(id)
    
    def step(self, obs) -> bool:
        return False