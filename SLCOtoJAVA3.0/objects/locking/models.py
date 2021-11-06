from objects.ast.models import VariableRef


class LockRequest:
    """A lock request for a specific variable and/or index."""
    def __init__(self, target: VariableRef):
        self.target = target
        self.id = -1

    def __repr__(self) -> str:
        return str(self.target)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, LockRequest):
            return self.target == o.target
        return False

    def __hash__(self) -> int:
        # A weak hash only containing the target variable is required, since indices are too complex to hash reliably.
        return hash(self.target)
