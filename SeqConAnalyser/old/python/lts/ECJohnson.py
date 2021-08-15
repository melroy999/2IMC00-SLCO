# The elementary circuit detection algorithm presented by Johnson.
from lts.SCCTarjan import identifySCCs


def find_sccs(dependency_graph):
    sscs = []
    states = set(dependency_graph.keys())
    identifySCCs(dependency_graph, states, {}, sscs)

    result = []
    for ssc in sscs:
        if ssc[0] <= 1:
            continue
        else:
            result.append(ssc[1])
    return result


class Vertex:
    """
    This class represents a vertex of a graph.
    It can be blocked as defined by Johnson's algorithm to find the elementary circuits.
    """

    def __init__(self, id):
        self.id = id
        self.is_blocked = False
        self.blocks = set()

    def __eq__(self, other):
        return self.id == other.id if isinstance(self, other.__class__) else False

    def __ne__(self, other):
        return not self.__eq__(other) if isinstance(self, other.__class__) else True

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return str(self.id)

    def block_self(self):
        """Indicate that this vertex is blocked."""
        self.is_blocked = True

    def block_other(self, v):
        """Indicate that the provided vertex is blocked by this vertex."""
        self.blocks.add(v)

    def reset_block(self):
        """Resets the blocking caused by this vertex."""
        self.is_blocked = False
        self.blocks = set()

    def unblock(self):
        """Unblocks this vertex along with any other vertex that may have been blocked by this one."""
        self.is_blocked = False

        for v in self.blocks:
            if v.is_blocked:
                v.unblock()

        self.blocks.clear()


class Graph:
    """
    The class represents a graph in the elementary circuit algorithm presented by Johnson.
    """

    def __init__(self, dependency_graph):
        self.vertices = []
        self.edges_outgoing = {}
        self.edges_incoming = {}

        _vertices = {}

        for src, outgoing in dependency_graph.items():
            v = _vertices.get(src, Vertex(src))

            if src not in _vertices:
                # The source vertex is not yet defined, define it.
                _vertices[src] = v
                self.vertices.append(v)

                # Initialize its edge collections.
                self.edges_outgoing[v] = []
                self.edges_incoming[v] = []

            for tgt, labels in outgoing.items():
                w = _vertices.get(tgt, Vertex(tgt))

                if tgt not in _vertices:
                    # The target vertex is not yet defined, define it.
                    _vertices[tgt] = w
                    self.vertices.append(w)

                    # Initialize its edge collections.
                    self.edges_outgoing[w] = []
                    self.edges_incoming[w] = []

                # Define the edge in the graph.
                self.edges_outgoing[v].append(w)
                self.edges_incoming[w].append(v)

        self.vertices = sorted(self.vertices, key=lambda v: v.id)

    def __len__(self):
        return len(self.vertices)

    def as_dependency_graph(self):
        """Returns this graph as a dependency graph structure. However, the transition labels are lost."""
        return {v.id: {w.id: set() for w in self.edges_outgoing[v]} for v in self.vertices}

    def shallow_copy(self):
        """Returns a shallow copy of this graph. The references to the vertex instances are kept."""
        graph = Graph({})
        graph.vertices = list(self.vertices)
        graph.edges_outgoing = {key: list(value) for key, value in self.edges_outgoing.items()}
        graph.edges_incoming = {key: list(value) for key, value in self.edges_incoming.items()}
        return graph

    def start(self):
        """Returns the start vertex (smallest index) of the graph."""
        return self.vertices[0] if len(self.vertices) > 0 else None

    def reset_block(self):
        """Resets the block of all vertices in the graph."""
        for i in range(0, len(self)):
            self.vertices[i].reset_block()

    def remove_vertex(self, v):
        """Removes the vertex from the graph along with any edges from- and to that vertex."""
        self.vertices.remove(v)

        for w in self.edges_outgoing[v]:
            self.edges_incoming[w].remove(v)
        del self.edges_outgoing[v]

        for w in self.edges_incoming[v]:
            self.edges_outgoing[w].remove(v)
        del self.edges_incoming[v]

    def get_sccs(self):
        """Returns a collection of Graph instances that represent the strongly connected components of this graph."""
        sccs = find_sccs(self.as_dependency_graph())

        result = []
        for scc in sccs:
            graph = self.shallow_copy()

            for v in list(graph.vertices):
                # This vertex in the original graph is not part of the SCC. It must be removed.
                if v.id not in scc:
                    graph.remove_vertex(v)

            result.append(graph)
        return result

    def circuit(self, s):
        """Returns the circuits of the graph that start at the `s' vertex."""

        # Initializes the stack that keeps track of the currently traversed path.
        stack = []

        # Keeps track of the found circuits.
        found_circuits = []

        def _circuit(v):
            has_circuit = False

            stack.append(v)
            v.block_self()

            # Check every vertex that can be reached directly from the current vertex.
            for w in self.edges_outgoing[v]:
                if w is s:
                    # A circuit has been found.
                    found_circuits.append(stack[:] + [s])
                    has_circuit = True
                elif not w.is_blocked:
                    # If the next vertex is not blocked, continue recursively.
                    if _circuit(w):
                        # Somewhere on the path a circuit has been found.
                        has_circuit = True

            if has_circuit:
                v.unblock()
            else:
                for w in self.edges_outgoing[v]:
                    w.block_other(v)

            _ = stack.pop()
            return has_circuit

        # Call the recursive function.
        _ = _circuit(s)

        return found_circuits


def find_circuits(dependency_graph):
    """
    Finds and returns the collection of elementary circuits in the strongly connected dependency_graph. An elementary
    circuit is a sequence of vertices that together specify a sequential path through the dependency_graph such that the
    first and last vertex are the same.
    """

    def _find_circuits(graph):
        # The collection of circuits that have been found.
        found_circuits = []

        for scc in graph.get_sccs():
            if len(scc) <= 1:
                continue

            # Select a starting vertex.
            s = scc.start()

            # Remove the blockage of all the vertices in the graph.
            scc.reset_block()

            # Get the circuits in the graph.
            found_circuits += scc.circuit(s)

            # Remove the vertex from the graph and remove any edges to/from that vertex.
            scc.remove_vertex(s)

            # Check the sub graph again after removing the starting vertex.
            found_circuits += _find_circuits(scc)

        return found_circuits

    graph = Graph(dependency_graph)
    circuits = _find_circuits(graph)

    return [[vertex.id for vertex in circuit] for circuit in circuits]
