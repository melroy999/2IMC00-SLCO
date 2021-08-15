# The minimal hitting set algorithm presented by Murakami and Uno.


def is_empty(collection):
    """Returns true when the provided collection is empty, otherwise false is returned."""
    return False if collection else True


class HyperGraph:
    """This class represents a hyper graph used by the MHS algorithms presented by Murakami and Uno."""

    def __init__(self, edges):
        self.index = -1
        self.vertices = set()
        self.edges = []

        for edge in edges:
            _edge = set(edge)
            self.vertices |= _edge
            self.edges.append(_edge)

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1

        if len(self.edges) <= self.index:
            raise StopIteration
        else:
            return self.edges[self.index]

    def includes(self, vertex):
        """Returns the list of edges that contains the provided vertex."""
        return HyperGraph(filter(lambda edge: vertex in edge, self.edges))

    def uncov(self, subset):
        """Returns the list of edges that is uncovered by the provided subset of vertices."""
        return HyperGraph(filter(lambda edge: is_empty(edge & subset), self.edges))

    def crit(self, subset, vertex):
        """Returns the list of edges that are critical for the provided vertex."""
        return HyperGraph(filter(lambda edge: (edge & subset) == {vertex}, self.edges))

    def update_crit_uncov(self, v, subset, crit, uncov):
        """Returns a new (crit, uncov) pair after updating them."""
        _crit = {key: list(value) for key, value in crit.items()}
        _uncov = list(uncov)

        for edge in self.includes(v):
            for vertex in subset:
                if edge in _crit.get(vertex, []):
                    _crit[vertex].remove(edge)

            if edge in _uncov:
                _uncov.remove(edge)

                if not v in _crit:
                    _crit[v] = []

                _crit[v].append(edge)

        return (_crit, _uncov)

    def mmcs(self):
        """Runs the MMCS algorithm to find the collection of minimal hitting sets in this hyper graph."""
        mhss = []

        def _not_redundant(subset, v):
            for vertex in subset:
                if is_empty(self.crit(subset | {v}, vertex).edges):
                    return False
            return True

        def _mmcs(mhs, cand, crit, uncov):
            if is_empty(uncov):
                mhss.append(mhs)
                return

            edge = uncov[0]
            c = cand & edge
            cand -= c

            for v in c:
                (_crit, _uncov) = self.update_crit_uncov(v, mhs, crit, uncov)

                if _not_redundant(mhs, v):
                    _mmcs(mhs | {v}, cand, _crit, _uncov)

                cand |= {v}

        # Run the recursive MMCS algorithm with its initial values.
        _mmcs(
            mhs=set(),
            cand=set(self.vertices),
            crit={},
            uncov=list(self.edges)
        )

        return mhss
