//
//  seqcon-analyser.cpp
//  
//
//  Created by Anton Wijs on 11/05/2020.
//

#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>

using namespace std;

// Memory model
enum MM { TSO, PSO, ARM };
// Access type
enum AccessType { READ, WRITE };
// Type of edge in an Abstract Event Graph
enum edgeType { PREDGE, CMPEDGE };

// Macro to check set membership. Defining this as an inline function somehow leads to performance penalty.
#define set_contains(S, e) (S.find(e) != S.end())

// Macro to insert at end of vector
#define vector_insert(V, e) (V.insert(V.end(), e))

#define point_stack_max_size 6

// Template class to maintain a map of objects
// of type A_Type, each mapped to a unique ID.
template <class A_Type> class IndexedMap {
	private:
		map<A_Type, int> m;
		int counter;
	public:
		IndexedMap() {
			counter = 0;
		}

		// Store item, if not yet done, and return index in storage
		int insert(A_Type item) {
			auto it = m.find(item);
			if (it != m.end()) {
				return it->second;
			}
			else {
				m.insert(pair<A_Type, int>(item, counter));
				return counter++;
			}
		}

		// Find the given item, and return its index. If not present, return -1
		int find(A_Type item) {
			auto it = m.find(item);
			if (it != m.end()) {
				return it->second;
			}
			else {
				return -1;
			}
		}

		size_t size() {
			return counter;
		}
};

// Template class for a searchable vector of elements of type A_Type.
// A map is used for the searching of elements. 
template <class A_Type> class SearchableVector {
	private:
		map<A_Type, int> m;
		vector<A_Type> storage;
		int counter;
	public:
		SearchableVector() {
			counter = 0;
		}

		// Store item, if not yet done, and return index in storage
		int insert(A_Type item) {
			auto it = m.find(item);
			if (it == m.end()) {
				m.insert(pair<A_Type, int>(item, counter));
				vector_insert(storage, item);
				return counter++;
			}
			else {
				return it->second;
			}
		}

		// Find the given item, and return its index. If not present, return -1
		int find(A_Type item) {
			auto it = m.find(item);
			if (it != m.end()) {
				return it->second;
			}
			else {
				return -1;
			}
		}

		// Get the element at given index
		A_Type& get(int index) {
			return storage[index];
		}

		typedef typename vector<A_Type>::iterator VAiterator;

		VAiterator begin() {
			return storage.begin();
		}

		VAiterator end() {
			return storage.end();
		}

		int size() {
			return storage.size();
		}
};

// Class to maintain a relation between integers.
class Relation {
	protected:
		map<int, set<int>> r;
	public:
		void copy(Relation R) {
			r.clear();
			r.insert(R.begin(), R.end());
		}

		virtual void insert(int i, vector<int> J) {
			auto it = r.find(i);
			if (it == r.end()) {
				r.insert(pair<int, set<int>>(i, set<int>(J.begin(), J.end())));
			}
			else {
				::copy(J.begin(), J.end(), inserter(it->second, it->second.end()));
			}
		}

		virtual void insert(int i, set<int> J) {
			auto it = r.find(i);
			if (it == r.end()) {
				r.insert(pair<int, set<int>>(i, set<int>(J.begin(), J.end())));
			}
			else {
				::copy(J.begin(), J.end(), inserter(it->second, it->second.end()));
			}
		}

		virtual void insert(int i, int j) {
			auto it = r.find(i);
			if (it == r.end()) {
				r.insert(pair<int, set<int>>(i, { j }));
			}
			else {
				it->second.insert(j);
			}
		}

		void erase(int i) {
			r.erase(i);
		}

		bool contains(int i) {
			auto it = r.find(i);
			return (it != r.end());
		}

		bool are_related(int i, int j) {
			if (!contains(i)) {
				return false;
			}
			else {
				auto it = get(i);
				return set_contains(it->second, j);
			}
		}

		void unrelate(int i, int j) {
			if (contains(i)) {
				auto it = get(i);
				if (set_contains(it->second, j)) {
					it->second.erase(j);
				}
				if (it->second.empty()) {
					erase(i);
				}
			}
		}

		// Precondition: an entry exists for i in r
		map<int, set<int>>::iterator get(int i) {
			return r.find(i);
		}

		map<int, set<int>>::iterator begin() {
			return r.begin();
		}
		map<int, set<int>>::iterator end() {
			return r.end();
		}
};

// Class to maintain a relation between integers, with the related integers stored in a vector, as opposed to a set.
// Benefit: indices can be used to retrieve the integer at a given index from the vector associated to another integer
// Drawback: the structure is less suitable for checking whether a given integer is related to another integer
template <class A_Type> class VectorRelation {
	protected:
		vector<vector<A_Type>> r;
	public:
		VectorRelation(size_t size) {
			r.resize(size);
		}

		void insert(int i, vector<A_Type> J) {
			r[i].insert(r[i].end(), J.begin(), J.end());
		}

		void insert(int i, set<A_Type> J) {
			r[i].insert(r[i].end(), J.begin(), J.end());
		}

		void insert(int i, A_Type j) {
			r[i].insert(r[i].end(), j);
		}

		A_Type get_element(int i, int index) {
			return r[i][index];
		}

		vector<A_Type>& get(int i) {
			return r[i];
		}

		int size() {
			return r.size();
		}

		int get_size(int i) {
			return r[i].size();
		}
};

// Class to maintain a bidirectional relation between integers.
// Checking in the opposite direction of the relation can be done efficiently.
class BiRelation : public Relation {
	protected:
		map<int, set<int>> r_rev;
	public:
		typedef int value_type;

		void copy(BiRelation R) {
			r.clear();
			r_rev.clear();
			r.insert(R.begin(), R.end());
			r_rev.insert(R.rev_begin(), R.rev_end());
		}

		void insert(int i, vector<int> J) override {
			auto it = r.find(i);
			if (it == r.end()) {
				r.insert(pair<int, set<int>>(i, set<int>(J.begin(), J.end())));
			}
			else {
				::copy(J.begin(), J.end(), inserter(it->second, it->second.end()));
			}
			for (auto j : J) {
				it = r_rev.find(j);
				if (it == r_rev.end()) {
					r_rev.insert(pair<int, set<int>>(j, { i }));
				}
				else {
					it->second.insert(i);
				}
			}
		}

		void insert(int i, set<int> J) override {
			auto it = r.find(i);
			if (it == r.end()) {
				r.insert(pair<int, set<int>>(i, set<int>(J.begin(), J.end())));
			}
			else {
				::copy(J.begin(), J.end(), inserter(it->second, it->second.end()));
			}
			for (auto j : J) {
				it = r_rev.find(j);
				if (it == r_rev.end()) {
					r_rev.insert(pair<int, set<int>>(j, { i }));
				}
				else {
					it->second.insert(i);
				}
			}
		}

		void insert(int i, int j) override {
			auto it = r.find(i);
			if (it == r.end()) {
				r.insert(pair<int, set<int>>(i, { j }));
			}
			else {
				it->second.insert(j);
			}
			it = r_rev.find(j);
			if (it == r_rev.end()) {
				r_rev.insert(pair<int, set<int>>(j, { i }));
			}
			else {
				it->second.insert(i);
			}
		}

		bool contains_rev(int i) {
			auto it = r_rev.find(i);
			return (it != r_rev.end());
		}

		// Precondition: an entry exists for i in r_rev
		map<int, set<int>>::iterator get_rev(int i) {
			return r_rev.find(i);
		}

		map<int, set<int>>::iterator rev_begin() {
			return r_rev.begin();
		}

		map<int, set<int>>::iterator rev_end() {
			return r_rev.end();
		}
};

// Struct to store info on Memory accesses
struct Access {
	int location; // Memory location
	bool local; // Is the variable local or not?
	AccessType type; // Type of access
	int ipos; // Pos of instructions of which the access is part
	vector<int> instr; // The IDs of the instruction of which the access is part
	int tid; // Thread executing the access
};

// < operator for Accesses, to be able to store them in a map
bool operator<(const Access& a1, const Access& a2) {
     return (a1.location<a2.location || (a1.location==a2.location && a1.local<a2.local)
     	|| (a1.location==a2.location && a1.local==a2.local && a1.type < a2.type)
     	|| (a1.location==a2.location && a1.local==a2.local && a1.type==a2.type && a1.ipos < a2.ipos)
     	|| (a1.location==a2.location && a1.local==a2.local && a1.type==a2.type && a1.ipos==a2.ipos && a1.tid < a2.tid));
}

// Struct to store LTS instruction label
struct Instruction {
	int pos; // Instruction position in the input model
	int tid; // SLCO state machine (thread) ID
	bool is_guarded; // Does the instruction have a condition?
	set<int> accesses; // Set of accesses performed by the instruction
	set<int> bottom_accs; // List of PR-smallest accesses
	vector<int> cond_reads; // List of reads to check a condition, in addition to the ones in bottom_accs
							// (the reads in cond_reads are address dependent on some in bottom_accs)
	set<int> top_accs; // List of PR-largest accesses
};

// Struct to store an LTS state
struct State {
	vector<int> outgoing_instr; // (Sorted!) vector of instruction ids outgoing from the state
	vector<pair<int, int>> predecessors; // Vector of predecessor states (needed to construct PR-relation)
};

// Struct to store CMP lookup info (for an access, thread pair)
struct ThreadAccessRange {
	int tid;
	int accesses_begin;
	int accesses_end;
};

// Struct to store PR reachability information
struct PRreach_tuple {
	int tgt;
	bool unsafe_directed;
	bool unsafe_undirected;
};

// Struct to store an item for the stackframe (in the elementary circuit detection procedure)
struct StackItem {
	int aid;
	edgeType edge_type;
	int edge_index;
	int t_index;
	bool cycle_found;
	int loc_count;

	void init(int aid, int l) {
		this->aid = aid;
		this->edge_type = PREDGE;
		this->edge_index = -1;
		this->t_index = -1;
		this->cycle_found = false;
		this->loc_count = l;
	}

	void init_CMP(int aid, int l) {
		this->aid = aid;
		this->edge_type = CMPEDGE;
		this->edge_index = -1;
		this->t_index = 0;
		this->cycle_found = false;
		this->loc_count = l;
	}

	void print() {
		cout << "(" << aid << ", ";
		if (edge_type == PREDGE) {
			cout << "PR";
		}
		else {
			cout << "CMP";
		}
		cout << ", edge_index=" << edge_index << ", t_index=" << t_index << ", cycle_found=" << cycle_found << ", loc_count=" << loc_count << ")" << endl;
	}
};

// Copy functions
void copy(StackItem& n, StackItem m) {
	n.aid = m.aid;
	n.edge_type = m.edge_type;
	n.edge_index = m.edge_index;
	n.t_index = m.t_index;
	n.cycle_found = m.cycle_found;
	n.loc_count = m.loc_count;
}

void copy(int& n, int m) {
	n = m;
}

void print(int n) {
	cout << n << endl;
}

// Struct to store an item on the processing stack when placing fences
struct StackItem_fencing {
	int instruction;
	vector<int> next;
	set<int> essential_instr;
};

// Struct to store a fence candidate, when placing fences
struct fence_candidate_t {
	int start; // first instruction in the path to be fenced
	bool Acum_needed; // is an A_cumulative fence required?
	set<pair<int, bool>> instr; // set of instructions in the path
};

// < operator for fence candidates, to be able to store them in a set
bool operator<(const fence_candidate_t& a1, const fence_candidate_t& a2) {
     return (a1.start < a2.start || (a1.start==a2.start && a1.Acum_needed < a2.Acum_needed)
     	|| (a1.start == a2.start && a1.Acum_needed == a2.Acum_needed && a1.instr < a2.instr));
}

void copy(StackItem_fencing& n, StackItem_fencing m) {
	n.instruction = m.instruction;
	n.next = m.next;
	n.essential_instr = m.essential_instr;
}

// Template class for a stack with a fixed preallocated space
template <class A_Type> class StaticStack {
	private:
		vector<A_Type> stack;
		size_t top;
	public:
		StaticStack(size_t s) {
			stack.resize(s);
			top = -1;
		}

		// Precondition: the stack is not empty
		A_Type& peek() {
			return stack[top];
		}

		// Precondition: there are at least two elements on the stack
		A_Type& peek_below_top() {
			return stack[top-1];
		}

		bool empty() {
			return (top == -1);
		}

		// Precondition: the stack is not empty
		void pop() {
			top--;
		}

		void push(A_Type n) {
			top++;
			copy(stack[top], n);
		}

		size_t size() {
			return top+1;
		}

		typename vector<A_Type>::iterator begin() {
			return stack.begin();
		}

		typename vector<A_Type>::iterator end() {
			return stack.begin() + top + 1;
		}

		typename vector<A_Type>::reverse_iterator rbegin() {
			return stack.rend() - 1 - top;
		}

		typename vector<A_Type>::reverse_iterator rend() {
			return stack.rend();
		}

		void print() {
			cout << "Stack content:" << endl;
			for (int i = 0; i < top+1; i++) {
				stack[i].print();
			}
			cout << "end stack content" << endl;
		}
};

// Accesses. Global structure, to make it accessible by the comparison function for sorting
SearchableVector<Access> accesses;

// Comparison function for sorting indices to accesses
bool compare_access_indices(int i, int j) {
	return accesses.get(i).tid < accesses.get(j).tid;
}

// Comparison function for sorting (access ID, bool) pairs. First give priority on access ID (descending order), and then on Boolean flag (true > false)
bool compare_access_bool_pairs(pair<int, bool> p1, pair<int, bool> p2) {
	if (p1.first > p2.first) {
		return true;
	}
	if (p2.first > p1.first) {
		return false;
	}
	else return p1.second;
}

// Function to check whether two accesses can be reordered
bool can_reorder(int ai, int bi, MM mmodel, Relation DP, Relation CTRL) {
	Access& a = accesses.get(ai);
	Access& b = accesses.get(bi);
	if (a.type == WRITE) {
		if (b.type == WRITE) {
			if (a.location == b.location) {
				return false;
			}
			else {
				return !(mmodel == TSO);
			}
		}
		else {
			return true;
		}
	}
	else {
		if (b.type == WRITE) {
			if (mmodel != ARM) {
				return false;
			}
			else {
				if (DP.are_related(bi, ai)) {
					return false;
				}
				else return !CTRL.are_related(bi, ai);
			}
		}
		else {
			if (mmodel != ARM) {
				return false;
			}
			else {
				return !(a.location == b.location || DP.are_related(bi, ai));
			}
		}
	}
}

// Function to PPR-reorder a given access ai, under the given relations, into the given instruction.
// Precondition: if ai stems from another instruction, then it is not yet reordered into instruction instr_id.
// Postcondition: ai is properly placed in the PPR-relation.
bool reorder(int ai, int instr_id, map<int, Instruction>& instructions, MM mmodel,
				vector<BiRelation>& PPR, BiRelation PR, BiRelation PRplus_intra_instr, Relation DP, Relation CTRL, bool from_outside_instr) {
	// Get the PPR-predecessors of ai.
	bool moved_into = false;
	auto instr = (instructions.find(instr_id));
	set<int> reordered;
	// if (ai == 37) {
	// 	cout << "reordering " << ai << endl;
	// }
	if ((!from_outside_instr && PRplus_intra_instr.contains_rev(ai)) || (from_outside_instr && !(instr->second.top_accs.empty()))) {
		set<int> set1, set2;
		set<int>& open = set1;
		set<int>& tmp = set1;
		set<int>& next = set2;

		if (!from_outside_instr) {
			auto it_pred = PR.get_rev(ai);
			for (auto bi : it_pred->second) {
				if (set_contains(instr->second.accesses, bi)) {
					open.insert(bi);
				}
			}
		}
		else {
			for (int bi : instr->second.top_accs) {
				open.insert(bi);
			}
		}
		while (!open.empty()) {
			for (int bi : open) {
				// Only consider bi if all its PPR-successors are in reordered.
				// Say we have bi -PR-> ci, bi -PR-> di, ci -PR-> ai and di -PR-> ai, then we may only consider reordering ai before bi if ai can be reordered
				// both before ci and di.
				bool check = true;
				if (PPR[instr_id].contains(bi)) {
					auto it = PPR[instr_id].get(bi);
					for (int ci : it->second) {
						if (!set_contains(reordered, ci)) {
							check = false;
							break;
						}
					}
				}
				if (check) {
					if (can_reorder(bi, ai, mmodel, DP, CTRL)) {
						reordered.insert(bi);
						if (!from_outside_instr) {
							if (!set_contains(instr->second.bottom_accs, bi) && PR.contains_rev(bi)) {
								auto it = PR.get_rev(bi);
								for (auto ci : it->second) {
									if (set_contains(instr->second.accesses, ci)) {
										next.insert(ci);
									}
								}
							}
						}
						else {
							if (PPR[instr_id].contains_rev(bi)) {
								auto it = PPR[instr_id].get_rev(bi);
								next.insert(it->second.begin(), it->second.end());
							}
							if (!moved_into) {
								instr->second.accesses.insert(ai);
								moved_into = true;
							}
						}
					}
					else {
						// ai cannot be reordered before bi, but maybe it can be, together with bi, before bi's PR-predecessors (those that are not PPR-predecessors of bi,
						// since we require that bi can also be reordered before them).
						//cout << "cannot reorder " << ai << " and " << bi << endl;
						PPR[instr_id].insert(bi, ai);
						if (!from_outside_instr) {
							if (PRplus_intra_instr.contains_rev(bi)) {
								map<int, set<int>>::iterator it_bi_pr_pred = PR.get_rev(bi);
								map<int, set<int>>::iterator it_bi_ppr_pred;
								bool has_PPR_preds = false;
								if (PPR[instr_id].contains_rev(bi)) {
									it_bi_ppr_pred = PPR[instr_id].get_rev(bi);
									has_PPR_preds = true;
								}
								for (int ci : it_bi_pr_pred->second) {
									if (set_contains(instr->second.accesses, ci)) {
										if (!has_PPR_preds || !set_contains(it_bi_ppr_pred->second, ci)) {
											next.insert(ci);
										}
									}
								}
							}
						}
					}
				}
				else {
					// ai and bi cannot be reordered
					if (moved_into) {
						PPR[instr_id].insert(bi, ai);
					}
				}
			}
			// Swap the sets
			open = next;
			next = tmp;
			tmp = open;
			next.clear();
		}
	}
	else if (from_outside_instr) {
		if (!set_contains(instr->second.accesses, ai)) {
			moved_into = true;
			instr->second.accesses.insert(ai);
		}
	}
	else {
		moved_into = true;
	}
	if (moved_into) {
		// If ai ends up not having PPR-predecessors, it is a new bottom access
		if (!PPR[instr_id].contains_rev(ai) && (from_outside_instr || PR.contains_rev(ai))) {
			instr->second.bottom_accs.insert(ai);
		}
		// If ai ends up not having PPR-successors, it is a new top access
		if (!PPR[instr_id].contains(ai) && (from_outside_instr || PR.contains(ai))) {
			instr->second.top_accs.insert(ai);
			// PPR-predecessors are not top accesses
			if (PPR[instr_id].contains_rev(ai)) {
				auto it = PPR[instr_id].get_rev(ai);
				for (int bi : it->second) {
					instr->second.top_accs.erase(bi);
				}
			}
		}
	}
	return moved_into;
}

// Function to iterate over the outgoing edges in an Abstract Event Graph of an access pointed to in StackItem s.
// The edges are stored in three vectors. This function simplifies iterating over all elements
// in those three separate vectors.
// Additional info provided to select edges:
// - initial_tid: the ID of the thread executing the first access selected (at the bottom of the search stack).
// - initial_ai: the initially selected access.
// - initial_loc_count: the number of times the initially chosen location has been visited.
// - visited_locs: set of locations that have previously been visited.
// - visited_threads: set of visited threads.
// - initial_ai_PR_explored: from initial_ai, a PR-path has been explored (this is relevant to know how a cycle can/must be closed).
// - unsafe_explored: at least one unsafe PR-path or CMP edge has been explored (hence a cycle may be produced).
// - PR_explored: at least one PR-path has been explored (another condition for a critical cycle).
// - atomicity_check: indicates whether atomicity checking should be performed.
// Returns the ID of the target access of the next edge.
// In addition, StackItem s has been updated to point to the selected edge. If no suitable edge exists, -1 is returned.
int get_next_edge(StackItem& s, int initial_ai, int& initial_loc_count, vector<bool>& visited_locs, int& number_of_locs_visited, vector<bool>& visited_threads,
						int& number_of_threads_visited, bool& initial_ai_PR_explored, int& unsafe_explored, int& PR_explored, bool atomicity_check,
						Relation RFE, vector<vector<pair<int, bool>>> PRedges, Relation PRplus_unsafe, VectorRelation<ThreadAccessRange> CMPt, VectorRelation<int> CMP) {
	int result = -1;
	int selected;
	int initial_tid = accesses.get(initial_ai).tid;
	// Consider a PR-path
	if (s.edge_type == PREDGE) {
		// Special case: if we are constructing a cycle visiting just one location, with initially a PR-path, then we are not allowed to select a PR-path.
		// In other words, we have constructed a path a -PR-> b -CMP-> c so far, a, b, c all accessing the same location, and only c -CMP-> a can be selected
		// (not relevant when atomicity checking is not done, as cycles with one location are not interesting).
		Access& a = accesses.get(s.aid);
		Access& a_initial = accesses.get(initial_ai);
		if (!(a.location == a_initial.location && initial_ai_PR_explored && s.loc_count == 3)) {
			for (int i = s.edge_index+1; i < PRedges[s.aid].size(); i++) {
				pair<int, bool> p = PRedges[s.aid][i];
				selected = p.first;
				if (selected < initial_ai) {
					// Jump to CMP-edges. (Based on the fact that reachable accesses are sorted by ID)
					break;
				}
				else {
				// if (selected >= initial_ai) {
					Access& b = accesses.get(selected);
					bool edge_is_unsafe = PRplus_unsafe.are_related(s.aid, selected) || PRplus_unsafe.are_related(selected, s.aid);
					// Either we are considering a thread other than the first one, or we have only selected one thread so far (a cycle cannot be closed yet),
					// or we must close a cycle. In that case, this step should lead to the initially selected access, and we must have explored at least one unsafe element.
					if (a.tid != initial_tid || number_of_threads_visited == 1 || (selected == initial_ai && (edge_is_unsafe || unsafe_explored > 0))) {
						// Either we are atomicity checking, or, if not, the locations of the previous and next access are not the same (see Don't Sit On The Fence)
						if (atomicity_check || a.location != b.location) {
							// Cycle closing location condition: if we are revisiting the initial location, then we must keep selecting that location to close the cycle.
							// If not, then we may either select a.location again, if s.loc_count allows it, or we can select a previously unvisited location.
							if ( (a.location == a_initial.location && number_of_locs_visited > 1 && s.loc_count < 3 && b.location == a_initial.location) ||
								 ( !(a.location == a_initial.location && number_of_locs_visited > 1) &&
								 		(((s.loc_count < 3 && a.location == b.location) || !visited_locs[b.location])
								 			|| (initial_loc_count < 3 && b.location == a_initial.location)) )
								) {
								s.edge_index = i;
								result = selected;
								break;
							}
						}
					}
				}
			}
		}
		if (result == -1) {
			s.edge_type = CMPEDGE;
			s.edge_index = -1;
			s.t_index = 0;
		}
	}
	// Consider a CMP-edge
	if (s.edge_type == CMPEDGE) {
		// cout << "considering CMP" << endl;
		vector<ThreadAccessRange>& out = CMPt.get(s.aid);
		for (int i = s.t_index; i < out.size(); i++) {
			// cout << "thread: " << out[i].tid << endl;
			if (!visited_threads[out[i].tid] || out[i].tid == initial_tid) {
				// cout << "not visited" << endl;
				// cout << s.t_index << endl;
				// cout << s.edge_index << endl;
				for (int j = (i == s.t_index ? s.edge_index+1 : out[i].accesses_begin); j < out[i].accesses_end; j++) {
					selected = CMP.get_element(s.aid, j);
					// cout << "considering " << selected << endl;
					if (selected >= initial_ai) {
						// cout << "CMP: checking " << selected << endl;
						Access& a = accesses.get(s.aid);
						Access& b = accesses.get(selected);
						Access& a_initial = accesses.get(initial_ai);
						// If we return to the initial thread, then we can either select an access different from the initial one (if we did not at the
						// start explore a PR-path of that thread), or we can select the initial access, by which we close a cycle.
						// In that case, the cycle should be critical, i.e., a PR-path must have been explored, and either an unsafe element must have been explored,
						// or the selected CMP-edge is unsafe.
						// cout << "here1" << endl;
						if (out[i].tid != initial_tid || (selected != initial_ai && !initial_ai_PR_explored) ||
								(selected == initial_ai && PR_explored > 0 && (unsafe_explored > 0 || RFE.are_related(s.aid, j) || RFE.are_related(j, s.aid)))) {
							// If we are not doing atomicity checking, single-location cycles are not interesting. If we are checking atomicity, we must select the initial access.
						// cout << "here2" << endl;
							if (a.location == a_initial.location && initial_ai_PR_explored && s.loc_count == 3 && (selected != initial_ai || !atomicity_check)) {
								continue;
							}
						// cout << "here3" << endl;
							// If we are revisiting the initial location, we must keep selecting accesses to that location.
							if (a.location == a_initial.location && number_of_locs_visited > 1 && b.location != a_initial.location) {
								continue;
							}
						// cout << "here4" << endl;
							// Select a valid location
							if ((s.loc_count < 3 && a.location == b.location) || (initial_loc_count < 3 && b.location == a_initial.location) ||
									selected == initial_ai || !visited_locs[b.location]) {
						// cout << "here5" << endl;
								s.edge_index = j;
								s.t_index = i;
								result = selected;
								return result;
							}
						}
					}
				}
			}
		}
	}
	return result;
}

// Function to update PR and PRinstr relations for two given instruction ids
void update_PR_PRinstr(int i, int j, BiRelation& PR, BiRelation& PRinstr, map<int, Instruction> instructions) {
	Instruction& s_instr = instructions.find(i)->second;
	Instruction& t_instr = instructions.find(j)->second;
	for (auto ta : s_instr.top_accs) {
		PR.insert(ta, t_instr.bottom_accs);
	}
	PRinstr.insert(i, j);
}

int main (int argc, char *argv[]) {
	string modelname = "";
	MM weakmemmodel = TSO;
	bool check_atomicity = false;
	bool static_analysis = false;
	bool verbose = false;

	if (argc < 2) {
		cout << "Model name missing!" << endl;
		exit(-1);
	}
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-w") == 0) {
			i++;
			if (strcmp(argv[i], "TSO") == 0) {
				weakmemmodel = TSO;
			}
			else if (strcmp(argv[i], "PSO") == 0) {
				weakmemmodel = PSO;
			}
			else if (strcmp(argv[i], "ARM") == 0) {
				weakmemmodel = ARM;
			}
			else if (strcmp(argv[i], "POWER") == 0) {
				weakmemmodel = ARM;
			}
			else {
				weakmemmodel = TSO;
			}
		}
		else if (strcmp(argv[i], "-a") == 0) {
			check_atomicity = true;
			cout << "Atomicity checking will be performed!" << endl;
		}
		else if (strcmp(argv[i], "-s") == 0) {
			static_analysis = true;
		}
		else if (strcmp(argv[i], "-v") == 0) {
			verbose = true;
		}
		else if (strcmp(argv[i], "-h") == 0) {
			cout << "Usage: seqcon-analyser [-wsa] model" << endl;
			cout << "" << endl;
			cout << "Check for sequentially inconsistent behaviour in the given .aut file containing the state space of an SLCO model, unless '-s' is used." << endl;
			cout << " -w                                  weak memory model to consider (TSO,PSO,ARM)  (default: TSO)" << endl;
			cout << " -s                                  apply only static analysis (ignore state space) (default: no)" << endl;
			cout << "                                       -> this option requires a .instructions file listing the instructions of the SLCO model" << endl;
			cout << " -a                                  apply atomicity checking in combination with SC checking (default: no)" << endl;
			cout << " -v                                  verbose mode;" << endl;
		}
		else {
			modelname = string(argv[i]);
		}
	}
	cout << "Checking input model under ";
	if (weakmemmodel == TSO) {
		cout << "TSO";
	}
	else if (weakmemmodel == PSO) {
		cout << "PSO";
	}
	else {
		cout << "ARM";
	}
	cout << endl;

	// Set static analysis to true in case a .instructions file is given
	if (modelname.find(".instructions") != string::npos) {
		static_analysis = true;
	}
	if (static_analysis) {
		cout << "Analysis of statically derived information is performed!" << endl;
	}

	// PR relation
	BiRelation PR;
	// Transitive closure of PR
	Relation PRplus;
	// Transitive closure of PR per instruction
	BiRelation PRplus_intra_instr;
	// PR relation at instruction level
	BiRelation PRinstr;
	// Dependency relation DP
	// Covers value and address dependencies, and RF for thread-local variables
	Relation DP;
	// CTRL dependency relation
	Relation CTRL;
	// Unsafe CMP relation, corresponding for ARM with RFE (external read from)
	Relation RFE;
	// Concurrency relation between instructions (dynamic IDs), constructed while reading the LTS, for efficient construction of CMP later on
	Relation CONC;
	
	// Various IndexedMaps to keep track of information extracted from the LTS
	IndexedMap<string> instruction_positions; // Maps short instruction position strings to integer IDs
	IndexedMap<string> instruction_ids; // Maps long instruction strings to integer IDs
	map<int, Instruction> instructions;
	IndexedMap<string> thread_ids;
	IndexedMap<string> location_ids;
	// Per thread, a vector of access IDs is stored executed by that thread
	vector<vector<int>> accesses_of_thread;

	// Reading the input LTS
	string line;
	ifstream inputfile (modelname);

	if (inputfile.is_open()) {
		size_t sep1, sep2;
		int nr_of_states, nr_of_trans;
		vector<State> lts_states;
		int current_state_index, current_trans_index, prev_src;
		bool first_trans;
		bool last_trans = false;
		if (!static_analysis) {
			// Header: extract number of states
			getline(inputfile, line);
			sep1 = line.find_last_of(",");
			nr_of_states = stoi(line.substr(sep1+1, line.find_last_of(")")-(sep1+1)));
			// Create vector for states
			lts_states.resize(nr_of_states);
			sep2 = line.find_first_of(",");
			nr_of_trans = stoi(line.substr(sep2+1, sep1-(sep2+1)));

			current_state_index = -1;
			prev_src = -1;
		}
		// Instruction instance to be used to create new instructions
		Instruction instr;
		// Access instance to be used to create new accesses
		Access acc;
		string label;
		int src, tgt;
		int iid;
		vector<pair<int, int>> succs_to_process_for_PR;
		// Set of closed states
		set<int> closed;
		while (getline(inputfile, line)) {
			if (static_analysis && line.substr(0,3) == "ST'") {
				// We are analysing statically derived information, and are about to construct the PR-relation at instruction level
				sep1 = line.find_first_of(" ");
				string from = line.substr(0, sep1);
				string to = line.substr(sep1+1, line.length()-(sep1+1));
				src = instruction_positions.find(from);
				tgt = instruction_positions.find(to);
				PRinstr.insert(src, tgt);
			}
			else {
				if (!static_analysis) {
					// Extract info from transition label
					sep1 = line.find_first_of(",");
					src = stoi(line.substr(1,sep1-1));
					if (src != prev_src) {
						first_trans = true;
						current_state_index++;
						last_trans = true;
					}
					else {
						first_trans = false;
					}
					sep2 = line.find_first_of("\"", sep1+2);
					label = line.substr(sep1+2, sep2-(sep1+2));
					tgt = stoi(line.substr(sep2+2, line.length()-(sep2+2)-1));
				}
				else {
					label = line;
				}
				// List of previous and current accesses, used to build intra-instruction PR-relation
				vector<int> prev_accesses, curr_accesses_bottom, curr_accesses_top;

				// Store long instruction string description
				iid = instruction_ids.find(label);
				// Check if instruction is already stored. If not, create it
				if (iid == -1) {
					iid = instruction_ids.insert(label);
					if (verbose) {
						cout << iid << ": " << label << endl;
					}
					if (label.compare("tau") != 0) {
						//cout << label << endl;
						// Break label further down
						label = label.substr(3, label.length()-4);
						// cout << label << endl;
						sep1 = label.find_first_of(",");
						sep2 = label.find_first_of(",", sep1+1);
						string thread = label.substr(sep1+2, sep2-sep1-2);
						int tid = thread_ids.insert(thread);
						sep1 = label.find_first_of(",", sep2+1);
						string statement = label.substr(sep2+2, sep1-sep2-2);
						// cout << "here: " << statement << endl;
						int ipos = instruction_positions.insert(statement);

						// Set the instruction info
						instr.pos = ipos;
						instr.tid = tid;
						instr.is_guarded = false;
						instr.accesses.clear();
						instr.bottom_accs.clear();
						instr.cond_reads.clear();
						instr.top_accs.clear();

						sep2 = label.find_first_of("[", sep1);
						sep1 = label.find_last_of("]");
						// cout << sep2 << endl;
						// cout << sep1 << endl;
						string accs = label.substr(sep2, sep1-sep2+1);
						// cout << "Accs:" << endl;
						// cout << accs << endl;

						// List of previous and current accesses, used to build intra-instruction PR-relation
						prev_accesses.clear();
						curr_accesses_bottom.clear();
						curr_accesses_top.clear();

						bool first = true;

						while (accs.compare("") != 0) {
							sep1 = accs.find_first_of("'");
							if (sep1 == string::npos) {
								break;
							}
							// Find the matching closing ']' of the reads of the next access pattern,
							// indicating the end of the (next) list of read accesses
							int bracket_counter = 0;
							for (sep2 = sep1+2; sep2 < accs.length(); sep2++) {
								if (accs[sep2] == ']') {
									if (bracket_counter == 1) {
										break;
									}
									else {
										bracket_counter--;
									}
								}
								else if (accs[sep2] == '[') {
									bracket_counter++;
								}
							}
							//cout << "label: " << accs << endl;
							string reads = accs.substr(sep1+2, sep2-(sep1+1));
							//cout << "reads: " << reads << endl;
							// Find the next "[" and "]", the corresponding list of write accesses
							sep1 = accs.find_first_of("[", sep2);
							sep2 = accs.find_first_of("]", sep1);
							string writes = accs.substr(sep1, sep2+1-sep1);
							accs = accs.substr(sep2);
							//cout << "writes: " << writes << endl;

							bool reads_stored = false;

							while (true) {
								// cout << "Reads:" << endl;
								// cout << reads << endl;
								sep2 = reads.find_first_of("'");
								if (sep2 == string::npos) {
									break;
								}
								// Record the read access
								sep1 = reads.find_first_of(",", sep2);
								if (sep1 == string::npos) {
									sep1 = reads.length()-1;
								}
								if (reads[sep2-1] == 'c') {
									sep1--;
								}
								string read = reads.substr(sep2+2, sep1-(sep2+2));
								
								// In case we are using statically derived information, abstract away array index computations
								if (static_analysis) {
									size_t sep5 = read.find_first_of("(");
									if (sep5 != string::npos) {
										read = read.substr(0, sep5+1) + "*)";
									}
								}

								// Create and store read access
								int loc = location_ids.insert(read);
								acc.location = loc;
								// cout << loc << endl;
								// Is the variable thread-local? (encoded in name by the fact that ' occurs more than once)
								acc.local = count(read.begin(), read.end(), '\'') > 1;
								// cout << acc.local << endl;
								acc.type = READ;
								// cout << acc.type << endl;
								acc.ipos = ipos;
								// cout << acc.instruction << endl;
								acc.tid = tid;
								// cout << tid << endl;
								int aid = accesses.insert(acc);
								if (verbose) {
									cout << "read " << aid << " : " << read << ": " << label << endl;
								}
								instr.accesses.insert(aid);
								reads_stored = true;

								if (reads[sep2-1] != 'p') {
									vector_insert(curr_accesses_bottom, aid);
								}
								else {
									vector_insert(curr_accesses_top, aid);
								}

								if (reads[sep2-1] == 'p') {
									// we have a tuple with a read and a list of address dependencies of that read
									// record the dependencies
									size_t sep3 = sep1+3;
									while (true) {
										size_t sep4 = reads.find_first_of(",]", sep3);
										string depread = reads.substr(sep3, sep4-sep3);
										// In case we are using statically derived information, abstract away array index computations
										if (static_analysis) {
											size_t sep5 = depread.find_first_of("(");
											if (sep5 != string::npos) {
												depread = depread.substr(0, sep5+1) + "*)";
											}
										}
										//cout << "depread: " << depread << endl;
										// Store this read access
										loc = location_ids.insert(depread);
										acc.location = loc;
										acc.local = count(depread.begin(), depread.end(), '\'') > 1;
										int depaid = accesses.insert(acc);

										// Store dependencies
										if (weakmemmodel == ARM) {
											DP.insert(aid, depaid);
										}
										PR.insert(depaid, aid);
										// This is an intra-instruction PR-connection. Store it as well in PRplus_instra_instr
										PRplus_intra_instr.insert(depaid, aid);

										sep3 = sep4+2;
										if (reads[sep4] == ']') {
											break;
										}
									}
									reads = reads.substr(sep3, reads.length()-sep3);
									//cout << "reads: " << reads << endl;
								}
								else {
									reads = reads.substr(sep1, reads.length()-sep1);
								}
							}
							// The instruction is guarded if there are no writes at this point
							if (writes.compare("[]") == 0) {
								instr.is_guarded = true;
								// Store additional reads to check the condition (besides those in bottom_accs), if needed
								if (!curr_accesses_top.empty()) {
									instr.cond_reads = curr_accesses_top;
								}
							}
							if (!curr_accesses_bottom.empty()) {
								// If needed, update PR
								if (!prev_accesses.empty()) {
									for (auto a : prev_accesses) {
										PR.insert(a, curr_accesses_bottom);
										// This is an intra-instruction PR-connection. Store it as well in PRplus_instra_instr
										PRplus_intra_instr.insert(a, curr_accesses_bottom);
									}
								}
								// Store PR-smallest accesses
								if (first) {
									first = false;
									instr.bottom_accs.insert(curr_accesses_bottom.begin(), curr_accesses_bottom.end());
								}
								// Swap lists of accesses
								if (!curr_accesses_top.empty()) {
									prev_accesses = curr_accesses_top;
									curr_accesses_top.clear();
								}
								else {
									prev_accesses = curr_accesses_bottom;
								}
								curr_accesses_bottom.clear();
							}

							while (true) {
								// cout << "Writes:" << endl;
								// cout << writes << endl;
								sep1 = writes.find_first_of("'");
								if (sep1 == string::npos) {
									break;
								}
								sep2 = writes.find_first_of(",");
								if (sep2 == string::npos) {
									sep2 = writes.length()-1;
								}
								sep2--;
								string write = writes.substr(sep1+2, sep2-(sep1+2));

								// In case we are using statically derived information, abstract away array index computations
								if (static_analysis) {
									size_t sep5 = write.find_first_of("(");
									if (sep5 != string::npos) {
										write = write.substr(0, sep5+1) + "*)";
									}
								}

								// Create and store write access
								int loc = location_ids.insert(write);
								acc.location = loc;
								// cout << loc << endl;
								// Is the variable thread-local? (encoded in name by the fact that ' occurs more than once)
								acc.local = count(write.begin(), write.end(), '\'') > 1;
								// cout << acc.local << endl;
								acc.type = WRITE;
								// cout << acc.type << endl;
								acc.ipos = ipos;
								// cout << acc.instruction << endl;
								acc.tid = tid;
								// cout << tid << endl;
								int aid = accesses.insert(acc);
								if (verbose) {
									cout << "write " << aid << " : " << write << ": " << label << endl;
								}
								instr.accesses.insert(aid);
								vector_insert(curr_accesses_bottom, aid);

								//cout << write << endl;
								writes = writes.substr(sep2, writes.length()-sep2);
								// cout << "here" << endl;
							}
							// cout << accs << endl;

							if (!curr_accesses_bottom.empty()) {
								// If needed, update PR
								if (!prev_accesses.empty()) {
									for (auto a : prev_accesses) {
										PR.insert(a, curr_accesses_bottom);
										// This is an intra-instruction PR-connection. Store it as well in PRplus_instra_instr
										PRplus_intra_instr.insert(a, curr_accesses_bottom);
									}
									// Writes depend on directly preceding reads
									if (reads_stored and weakmemmodel == ARM) {
										for (auto a : curr_accesses_bottom) {
											DP.insert(a, prev_accesses);
										}
									}
								}
								// Store PR-smallest accesses
								if (first) {
									first = false;
									instr.bottom_accs.insert(curr_accesses_bottom.begin(), curr_accesses_bottom.end());
								}
								// Swap lists of accesses
								prev_accesses = curr_accesses_bottom;
								curr_accesses_bottom.clear();
							}
						}
						// Store PR-largest accesses
						instr.top_accs.insert(prev_accesses.begin(), prev_accesses.end());
					}
					else {
						// Store a dummy tau instruction
						instr.tid = -1;
						instr.accesses.clear();
					}
					// Store the instruction
					auto it = instructions.insert(pair<int, Instruction>(iid, instr));
				}
			}
			if (!static_analysis) {
				// Get the instruction
				auto it = instructions.find(iid);
				// Store the source state info: outgoing instruction
				vector_insert(lts_states[current_state_index].outgoing_instr, iid);

				// Update PR
				if (last_trans) {
					// Compare outgoing transitions with those of the states in the todo list.
					// First sort the vector of outgoing instruction ids
					if (prev_src != -1) {
						State& s = lts_states[prev_src];
						sort(s.outgoing_instr.begin(), s.outgoing_instr.end());
						for (pair<int, int> p : succs_to_process_for_PR) {
							int i = 0;
							int j = 0;
							State&t = lts_states[p.second];
							while (i < t.outgoing_instr.size()) {
								int t_instr_id = t.outgoing_instr[i];
								if (j < s.outgoing_instr.size()) {
									int s_instr_id = s.outgoing_instr[j];
									if (s_instr_id < t_instr_id) {
										j++;
									}
									else if (s_instr_id == t_instr_id) {
										// if (t_instr_id == p.first) {
										// 	// p.first is PR after itself
										// 	// add t_instr_id to PR
										// 	// PR-relate top elements of instr with bottom elements of tgt_instr
										// 	update_PR_PRinstr(p.first, t_instr_id, PR, PRinstr, instructions);
										// }
										i++;
										j++;
									}
									else {
										// t_instr_id < s_instr_id
										// add t_instr_id to PR if p.first and t_instr_id stem from the same thread
										update_PR_PRinstr(p.first, t_instr_id, PR, PRinstr, instructions);
										i++;
									}
								}
								else {
									// add t_instr_id to PR
									update_PR_PRinstr(p.first, t_instr_id, PR, PRinstr, instructions);
									i++;
								}
							}
						}
						succs_to_process_for_PR.clear();
						// Process the predecessors in a similar way as the items on the todo list
						for (pair<int, int> p : s.predecessors) {
							int i = 0;
							int j = 0;
							State&t = lts_states[p.second];
							while (i < s.outgoing_instr.size()) {
								int s_instr_id = s.outgoing_instr[i];
								if (j < t.outgoing_instr.size()) {
									int t_instr_id = t.outgoing_instr[j];
									if (t_instr_id < s_instr_id) {
										j++;
									}
									else if (t_instr_id == s_instr_id) {
										// if (s_instr_id == p.first) {
										// 	// p.first is PR after itself
										// 	// add s_instr_id to PR
										// 	// PR-relate top elements of instr with bottom elements of tgt_instr
										// 	update_PR_PRinstr(p.first, s_instr_id, PR, PRinstr, instructions);
										// }
										i++;
										j++;
									}
									else {
										// s_instr_id < t_instr_id
										// add s_instr_id to PR if p.first and s_instr_id stem from the same thread
										update_PR_PRinstr(p.first, s_instr_id, PR, PRinstr, instructions);
										i++;
									}
								}
								else {
									// add s_instr_id to PR
									update_PR_PRinstr(p.first, s_instr_id, PR, PRinstr, instructions);
									i++;
								}
							}
						}
						// Update CONC relation
						for (int i = 0; i < s.outgoing_instr.size(); i++) {
							int i_id = s.outgoing_instr[i];
							Instruction& i_instr = instructions.find(i_id)->second;
							for (int j = i+1; j < s.outgoing_instr.size(); j++) {
								int j_id = s.outgoing_instr[j];
								Instruction& j_instr = instructions.find(j_id)->second;
								if (i_instr.tid != j_instr.tid) {
									if (i_id < j_id) {
										CONC.insert(i_id, j_id);
									}
									else {
										CONC.insert(j_id, i_id);
									}
								}
							}
						}
					}
					prev_src = src;
					last_trans = false;
				}
				if (set_contains(closed, tgt)) {
					vector_insert(succs_to_process_for_PR, (pair<int, int>(iid, tgt)));
				}
				else {
					vector_insert(lts_states[tgt].predecessors, (pair<int, int>(iid, src)));
				}
				closed.insert(prev_src);
				current_trans_index++;
			}
		}
		if (!static_analysis) {
			// Store info for final state
			// Compare outgoing instructions with those of the states in the todo list.
			// First sort the vector of outgoing instruction ids
			State& s = lts_states[src];
			sort(s.outgoing_instr.begin(), s.outgoing_instr.end());
			for (pair<int, int> p : succs_to_process_for_PR) {
				int i = 0;
				int j = 0;
				State&t = lts_states[p.second];
				while (i < t.outgoing_instr.size()) {
					int t_instr_id = t.outgoing_instr[i];
					if (j < s.outgoing_instr.size()) {
						int s_instr_id = s.outgoing_instr[j];
						if (s_instr_id < t_instr_id) {
							j++;
						}
						else if (s_instr_id == t_instr_id) {
							// if (t_instr_id == p.first) {
							// 	// p.first is PR after itself
							// 	// add t_instr_id to PR
							// 	// PR-relate top elements of instr with bottom elements of tgt_instr
							// 	update_PR_PRinstr(p.first, t_instr_id, PR, PRinstr, instructions);
							// }
							i++;
							j++;
						}
						else {
							// t_instr_id < s_instr_id
							// add t_instr_id to PR if p.first and t_instr_id stem from the same thread
							update_PR_PRinstr(p.first, t_instr_id, PR, PRinstr, instructions);
							i++;
						}
					}
					else {
						// add t_instr_id to PR
						update_PR_PRinstr(p.first, t_instr_id, PR, PRinstr, instructions);
						i++;
					}
				}
			}
			succs_to_process_for_PR.clear();
			// Process the predecessors in a similar way as the items on the todo list
			for (pair<int, int> p : s.predecessors) {
				int i = 0;
				int j = 0;
				State&t = lts_states[p.second];
				while (i < s.outgoing_instr.size()) {
					int s_instr_id = s.outgoing_instr[i];
					if (j < t.outgoing_instr.size()) {
						int t_instr_id = t.outgoing_instr[j];
						if (t_instr_id < s_instr_id) {
							j++;
						}
						else if (t_instr_id == s_instr_id) {
							// if (s_instr_id == p.first) {
							// 	// p.first is PR after itself
							// 	// add s_instr_id to PR
							// 	// PR-relate top elements of instr with bottom elements of tgt_instr
							// 	update_PR_PRinstr(p.first, s_instr_id, PR, PRinstr, instructions);
							// }
							i++;
							j++;
						}
						else {
							// s_instr_id < t_instr_id
							// add s_instr_id to PR if p.first and s_instr_id stem from the same thread
							update_PR_PRinstr(p.first, s_instr_id, PR, PRinstr, instructions);
							i++;
						}
					}
					else {
						// add s_instr_id to PR
						update_PR_PRinstr(p.first, s_instr_id, PR, PRinstr, instructions);
						i++;
					}
				}
			}
			// Update CONC relation
			for (int i = 0; i < s.outgoing_instr.size(); i++) {
				int i_id = s.outgoing_instr[i];
				Instruction& i_instr = instructions.find(i_id)->second;
				for (int j = i+1; j < s.outgoing_instr.size(); j++) {
					int j_id = s.outgoing_instr[j];
					Instruction& j_instr = instructions.find(j_id)->second;
					if (i_instr.tid != j_instr.tid) {
						if (i_id <= j_id) {
							CONC.insert(i_id, j_id);
						}
						else {
							CONC.insert(j_id, i_id);
						}
					}
				}
			}
		}
		// for (auto i : PR) {
		// 	for (auto j : i.second) {
		// 		cout << "(" << i.first << ", " << j << ")" << endl;
		// 	}
		// }
		inputfile.close();

		if (!static_analysis) {
			cout << "Finished reading LTS!" << endl;
			cout << "Number of states in LTS: " << nr_of_states << endl;
			cout << "Number of transitions in LTS: " << nr_of_trans << endl;
		}

		chrono::steady_clock::time_point begin = chrono::steady_clock::now();

		int cnt = 0;
		for (auto it : PR) {
			cnt += it.second.size();
		}

		// Construct the accesses of thread datastructure
		accesses_of_thread.resize(thread_ids.size());
		for (int ai = 0; ai < accesses.size(); ai++) {
			Access& a = accesses.get(ai);
			vector_insert(accesses_of_thread[a.tid], ai);
		}

		// For each access, create a list of instructions of which it is part
		for (auto p : instructions) {
			for (int ai : p.second.accesses) {
				vector_insert(accesses.get(ai).instr, p.first);
			}
		}

		// Compute transitive closure of PRplus_intra_instr using Floyd-Warshall
		for (auto i : instructions) {
			for (int ai : i.second.accesses) {
				for (int bi : i.second.accesses) {
					for (int ci : i.second.accesses) {
						if (PRplus_intra_instr.are_related(bi, ai) && PRplus_intra_instr.are_related(ai, ci)) {
							PRplus_intra_instr.insert(bi, ci);
						}
					}
				}
			}
		}

		// For each instruction, create an access reorder relation (PPR, subrelation of PR, consisting of the PR-pairs safe under the weak memory model).
		// Initially, this is empty.
		vector<BiRelation> PPR(instructions.size());

		// // if (!static_analysis) {
		// // 	// Set of (instruction position, thread id) pairs of outgoing transitions of a state
		// // 	set<pair<int, int>> out;
		// // 	// Compute inter-instruction PR and PRinstr
		// // 	for (auto s : lts_states) {
		// // 		out.clear();
		// // 		// Collect info on outgoing transitions
		// // 		for (int i = s.outgoing_begin; i < s.outgoing_end; i++) {
		// // 			auto instr = (*(lts_transitions[i].instruction)).second;
		// // 			out.insert(pair<int, int>(instr.pos, instr.tid));
		// // 		}
		// // 		// Check successors
		// // 		for (int i = s.outgoing_begin; i < s.outgoing_end; i++) {
		// // 			int tgt = lts_transitions[i].target;
		// // 			auto instr = (*(lts_transitions[i].instruction)).second;
		// // 			int instr_id = (*(lts_transitions[i].instruction)).first;
		// // 			int pos = instr.pos;
		// // 			int tid = instr.tid;
		// // 			for (int j = lts_states[tgt].outgoing_begin; j < lts_states[tgt].outgoing_end; j++) {
		// // 				int tgt_instr_id = (*(lts_transitions[j].instruction)).first;
		// // 				auto tgt_instr = (*(lts_transitions[j].instruction)).second;
		// // 				int tgt_pos = tgt_instr.pos;
		// // 				int tgt_tid = tgt_instr.tid;
		// // 				if (tgt_tid == tid) {
		// // 					if (out.find(pair<int, int>(tgt_pos, tgt_tid)) == out.end()) {
		// // 						PR-relate top elements of instr with bottom elements of tgt_instr
		// // 						for (auto ta : instr.top_accs) {
		// // 							PR.insert(ta, tgt_instr.bottom_accs);
		// // 							PRinstr.insert(instr_id, tgt_instr_id);
		// // 						}
		// // 					}
		// // 				}
		// // 			}
		// // 		}
		// // 	}
		// // }
		// else {
		if (static_analysis) {
			// We are working with statically derived information. Compute inter-instruction PR based on already read PRinstr
			for (int i = 0; i < instructions.size(); i++) {
				if (PRinstr.contains(i)) {
					Instruction& instr_i = instructions.find(i)->second;
					auto it = PRinstr.get(i);
					for (int j : it->second) {
						Instruction& instr_j = instructions.find(j)->second;
						for (int a : instr_i.top_accs) {
							PR.insert(a, instr_j.bottom_accs);
						}
					}
				}
			}
		}

		cout << "Number of accesses: " << accesses.size() << endl;
		int count = 0;
		for (auto it : PR) {
			count += it.second.size();
		}
		cout << "Number of PR-edges in AEG: " << count << endl;

		// Compute the RF relation for thread-local variables. This is integrated into DP.
		if (weakmemmodel == ARM) {
			set<int> open;
			closed.clear();
			for (auto i : PR) {
				Access& a = accesses.get(i.first);
				if (a.local && a.type == WRITE) {
					open.clear();
					closed.clear();
					closed.insert(i.first);
					// Search for reachable reads from the same location
					open.insert(i.second.begin(), i.second.end());
					while (!open.empty()) {
						int j = *(open.begin());
						open.erase(j);
						closed.insert(j);
						Access& b = accesses.get(j);
						if (a.location != b.location || b.type == READ) {
							if (a.location == b.location) {
								DP.insert(j, i.first);
							}
							if (PR.contains(j)) {
								auto it = PR.get(j);
								for (auto k : it->second) {
									if (closed.find(k) == closed.end()) {
										open.insert(k);
									}
								}
							}
						}
					}
				}
			}
		}

		// Compute PRplus via Floyd-Warshall
		PRplus.copy(PR);

		for (int k = 0; k < accesses.size(); k++) {
			if (PRplus.contains(k)) {
				int tid = accesses.get(k).tid;
				for (int i : accesses_of_thread[tid]) {
					if (PRplus.are_related(i, k)) {
						for (int j : accesses_of_thread[tid]) {
							if (PRplus.are_related(k, j)) {
								PRplus.insert(i, j);
							}
						}
					}
				}
			}
		}

		if (weakmemmodel == ARM) {
			// Using PRplus, compute the CTRL relation
			for (auto i : instructions) {
				if (i.second.is_guarded) {
					// The bottom accesses of i and those in cond_reads are the reads necessary to evaluate a condition
					for (auto ai : i.second.bottom_accs) {
						if (PRplus.contains(ai)) {
							auto it = PRplus.get(ai);
							for (auto aj : it->second) {
								CTRL.insert(aj, ai);
							}
						}
					}
					for (auto ai : i.second.cond_reads) {
						if (PRplus.contains(ai)) {
							auto it = PRplus.get(ai);
							for (auto aj : it->second) {
								CTRL.insert(aj, ai);
							}
						}
					}
				}
			}
		}

		// Remove self-loops and local variable accesses from PRplus
		for (int ai = 0; ai < accesses.size(); ai++) {
			if (PRplus.contains(ai)) {
				if (accesses.get(ai).local) {
					PRplus.erase(ai);
				}
				else {
					auto it = PRplus.get(ai);
					for (auto sit = it->second.begin(); sit != it->second.end();) {
						if (*sit == ai || accesses.get(*sit).local) {
							sit = it->second.erase(sit);
						}
						else {
							++sit;
						}
					}
				}
			}
		}

		// In each instruction, reorder accesses to obtain the PPR-relations
		set<int> set1, set2;
		set<int>& openset = set1;
		set<int>& tmpset = set1;
		set<int>& nextset = set2;
		for (auto instr : instructions) {
			openset.clear();
			nextset.clear();
			openset.insert(instr.second.bottom_accs.begin(), instr.second.bottom_accs.end());
			while (!openset.empty()) {
				for (int ai : openset) {
					reorder(ai, instr.first, instructions, weakmemmodel, PPR, PR, PRplus_intra_instr, DP, CTRL, false);
					if (PR.contains(ai) && PRplus_intra_instr.contains(ai)) {
						auto it = PR.get(ai);
						for (int bi : it->second) {
							if (set_contains(instr.second.accesses, bi)) {
								nextset.insert(bi);
							}
						}
					}
				}
				// Swap the sets
				openset = nextset;
				nextset = tmpset;
				tmpset = openset;
				nextset.clear();
			}
		}

		// cout << "The PPRs: " << endl;
		// for (int i = 0; i < instructions.size(); i++) {
		// 	cout << "For instruction " << i << ":" << endl;
		// 	for (auto i : PPR[i]) {
		// 		for (auto j : i.second) {
		// 			cout << "(" << i.first << ", " << j << ")" << endl;
		// 		}
		// 	}
		// }


		// cout << "here" << endl;
		// Now reorder accesses across instructions
		vector<vector<int>> accvector1(instructions.size());
		vector<vector<int>> accvector2(instructions.size());
		vector<vector<int>>& newly_added_accesses = accvector1;
		vector<vector<int>>& next_added_accesses = accvector2;
		vector<vector<int>>& tmp_accesses = accvector1;

		for (auto instr : instructions) {
			openset.clear();
			nextset.clear();
			openset.insert(instr.second.bottom_accs.begin(), instr.second.bottom_accs.end());
			while (!openset.empty()) {
				for (int ai : openset) {
					newly_added_accesses[instr.first].insert(newly_added_accesses[instr.first].end(), ai);
					if (PPR[instr.first].contains(ai)) {
						auto it = PPR[instr.first].get(ai);
						for (int bi : it->second) {
							if (!set_contains(nextset, bi)) {
								nextset.insert(bi);
								// cout << "adding " << bi << endl;
							}
						}
					}
				}
				openset = nextset;
				nextset = tmpset;
				tmpset = openset;
				nextset.clear();
			}
		}
		// cout << "here2" << endl;

		openset.clear();
		nextset.clear();
		for (int i = 0; i < instructions.size(); i++) {
			openset.insert(i);
		}
		while (!openset.empty()) {
			for (int instr_id : openset) {
				if (PRinstr.contains(instr_id)) {
					auto instr_it = PRinstr.get(instr_id);
					for (int instr_id2 : instr_it->second) {
						if (!newly_added_accesses[instr_id2].empty()) {
							for (int ai : newly_added_accesses[instr_id2]) {
								auto instr = (instructions.find(instr_id));
								if (instr->second.accesses.find(ai) == instr->second.accesses.end()) {
									bool result = reorder(ai, instr_id, instructions, weakmemmodel, PPR, PR, PRplus_intra_instr, DP, CTRL, true);
									if (result) {
										if (next_added_accesses[instr_id].empty()) {
											if (PRinstr.contains_rev(instr_id)) {
												auto instr_it2 = PRinstr.get_rev(instr_id);
												nextset.insert(instr_it2->second.begin(), instr_it2->second.end());
											}
										}
										next_added_accesses[instr_id].insert(next_added_accesses[instr_id].end(), ai);
									}
								}
							}
						}
					}
				}
			}
			// Swap
			openset = nextset;
			nextset = tmpset;
			tmpset = openset;
			nextset.clear();
			newly_added_accesses = next_added_accesses;
			next_added_accesses = tmp_accesses;
			tmp_accesses = newly_added_accesses;
			for (int i = 0; i < instructions.size(); i++) {
				next_added_accesses[i].clear();
			}
		}

		// CMP relation
		VectorRelation<int> CMP(accesses.size());
		// Unsafe CMP relation, corresponding for ARM with RFE (external read from)
		Relation RFE;
		// CMP at thread level (for a given access a provides the thread IDs of accesses that conflict with a, plus their respective begin and
		// end indices in the list of accesses CMP-conflicting with a)
		VectorRelation<ThreadAccessRange> CMPt(accesses.size());

		if (!static_analysis) {
			// Compute CMP, using the reordering information in the instructions and the CONC relation
			set<pair<int, int>> pairset;
			for (auto i_it : CONC) {
				int i = i_it.first;
				auto instr_i = instructions.find(i)->second;
				int tid_i = instr_i.tid;
				// Compare CONC-related instructions (of different threads)
				for (int j : i_it.second) {
					auto instr_j = instructions.find(j)->second;
					int tid_j = instr_j.tid;
					for (int ai : instr_i.accesses) {
						Access& a = accesses.get(ai);
						for (int bi : instr_j.accesses) {
							Access& b = accesses.get(bi);
							if (a.location == b.location && (a.type == WRITE || b.type == WRITE)) {
								if (ai < bi) {
									pairset.insert(pair<int, int>(ai, bi));
								}
								else {
									pairset.insert(pair<int, int>(bi, ai));									
								}
							}
						}
					}
				}
			}
			for (pair<int, int> p : pairset) {
				CMP.insert(p.first, p.second);
				CMP.insert(p.second, p.first);
			}
			pairset.clear();
		}
		else {
			// When working with statically derived information, CMP consists of all pairs of accesses of different threads that conflict
			for (int ai = 0; ai < accesses.size(); ai++) {
				Access& a = accesses.get(ai);
				for (int bi = ai+1; bi < accesses.size(); bi++) {
					Access& b = accesses.get(bi);
					if (a.tid != b.tid) {
						if (a.location == b.location && (a.type == WRITE || b.type == WRITE)) {
							CMP.insert(ai, bi);
							CMP.insert(bi, ai);
						}
					}
				}
			}
		}

		count = 0;
		for (int i = 0; i < CMP.size(); i++) {
			count += CMP.get(i).size();
		}
		cout << "Number of CMP-edges in AEG: " << count/2 << endl;

		// Sort the CMP vectors stored by the CMP relation by thread ID
		for (int ai = 0; ai < accesses.size(); ai++) {
			sort(CMP.get(ai).begin(), CMP.get(ai).end(), compare_access_indices);
		}

		// Transitively close the PPR relations, via Floyd-Warshall
		for (int w = 0; w < instructions.size(); w++) {
			Instruction& instr = instructions.find(w)->second;
			for (int k : instr.accesses) {
				for (int i : instr.accesses) {
					if (PPR[w].are_related(i, k)) {
						for (int j : instr.accesses) {
							if (PPR[w].are_related(k, j)) {
								PPR[w].insert(i, j);
							}
						}
					}
				}
			}
		}

		// For each access, create a vector containing int, bool tuples, with the int being IDs of accesses that are PR-reachable,
		// , and the bool indicating whether an PR-unsafe path exists in any of the two directions between the two accesses.
		// The bools are actually only used for the initial sorting of the vectors, not during cycle detection. For this, unsafety of PRpaths is derived
		// from the (updated during the procedure) relation PRplus_unsafe.
		vector<vector<pair<int, bool>>> reachable(accesses.size());
		// A Relation PRplus_unsafe to keep track of the unsafe PR-paths that still require analysis in cycle detection.
		Relation PRplus_unsafe;

		for (int ai = 0; ai < accesses.size(); ai++) {
			Access& a = accesses.get(ai);
			for (int bi : accesses_of_thread[a.tid]) {
				if (bi > ai) {
					bool unsafe_PRpath_exists_ai_to_bi = false;
					bool unsafe_PRpath_exists_bi_to_ai = false;
					bool relate_ai_bi = false;
					bool relate_bi_ai = false;
					// If ai and bi are related via PRplus, record their connection in reachable, but only if they are relevant for cycles
					if (PRplus.are_related(ai, bi) || PRplus.are_related(bi, ai)) {
						Access& b = accesses.get(bi);
						// In the case of atomicity checking, the presence of a PR-path between ai and bi inside a single instruction is sufficient for the presence of an unsafe PR-path
						if (a.ipos == b.ipos) {
							if (PRplus_intra_instr.are_related(bi, ai)) {
								if (check_atomicity) {
									unsafe_PRpath_exists_ai_to_bi = true;
									relate_ai_bi = true;
								}
								// Moving in the PR-direction inside instructions is only needed if a and b address different locations
								if (a.location != b.location) {
									relate_bi_ai = true;
									for (int b_instr : b.instr) {
										Instruction& instr = instructions.find(b_instr)->second;
										if (set_contains(instr.accesses, ai)) {
											if (!PPR[b_instr].are_related(bi, ai)) {
												unsafe_PRpath_exists_bi_to_ai = true;
												break;
											}
										}
									}
									if (!check_atomicity) {
										if (PRplus.are_related(ai, bi)) {
											relate_ai_bi = true;
											for (int a_instr : a.instr) {
												Instruction& instr = instructions.find(a_instr)->second;
												if (set_contains(instr.accesses, bi)) {
													if (!PPR[a_instr].are_related(ai, bi)) {
														unsafe_PRpath_exists_ai_to_bi = true;
														break;
													}
												}
											}
										}
									}
								}
							}
							else if (PRplus_intra_instr.are_related(ai, bi)) {
								if (check_atomicity) {
									unsafe_PRpath_exists_bi_to_ai = true;
									relate_bi_ai = true;
								}
								// Moving in the PR-direction inside instructions is only needed if a and b address different locations
								if (a.location != b.location) {
									relate_ai_bi = true;
									for (int a_instr : a.instr) {
										Instruction& instr = instructions.find(a_instr)->second;
										if (set_contains(instr.accesses, bi)) {
											if (!PPR[a_instr].are_related(ai, bi)) {
												unsafe_PRpath_exists_ai_to_bi = true;
												break;
											}
										}
									}
									if (!check_atomicity) {
										if (PRplus.are_related(bi, ai)) {
											relate_bi_ai = true;
											for (int b_instr : b.instr) {
												Instruction& instr = instructions.find(b_instr)->second;
												if (set_contains(instr.accesses, ai)) {
													if (!PPR[b_instr].are_related(bi, ai)) {
														unsafe_PRpath_exists_bi_to_ai = true;
														break;
													}
												}
											}
										}
									}
								}
							}
						}
						else {
							// We only need to consider a and b addressing different locations
							if (a.location != b.location) {
								if (PRplus.are_related(ai, bi)) {
									relate_ai_bi = true;
									// Is there an instruction in which both accesses occur (b has been reordered into the instruction of a)
									for (int a_instr : a.instr) {
										Instruction& instr = instructions.find(a_instr)->second;
										if (set_contains(instr.accesses, bi)) {
											if (!PPR[a_instr].are_related(ai, bi)) {
												unsafe_PRpath_exists_ai_to_bi = true;
												break;
											}
										}
									}
								}
								if (PRplus.are_related(bi, ai)) {
									relate_bi_ai = true;
									// Is there an instruction in which both accesses occur (a has been reordered into the instruction of b)
									for (int b_instr : b.instr) {
										Instruction& instr = instructions.find(b_instr)->second;
										if (set_contains(instr.accesses, ai)) {
											if (!PPR[b_instr].are_related(bi, ai)) {
												unsafe_PRpath_exists_bi_to_ai = true;
												break;
											}
										}
									}
								}
							}
						}
						// Record relation
						if (relate_ai_bi || relate_bi_ai) {
							vector_insert(reachable[ai], (pair<int, bool>(bi, unsafe_PRpath_exists_ai_to_bi || unsafe_PRpath_exists_bi_to_ai)));
							if (unsafe_PRpath_exists_ai_to_bi) {
								PRplus_unsafe.insert(ai, bi);
							}
							vector_insert(reachable[bi], (pair<int, bool>(ai, unsafe_PRpath_exists_ai_to_bi || unsafe_PRpath_exists_bi_to_ai)));
							if (unsafe_PRpath_exists_bi_to_ai) {
								PRplus_unsafe.insert(bi, ai);
							}
						}
						// cout << "PR path: " << ai << " -> " << bi << ": " << unsafe_PRpath_exists_ai_to_bi << endl;
						// cout << "PR path: " << bi << " -> " << ai << ": " << unsafe_PRpath_exists_bi_to_ai << endl;
					}
				}
			}
			// Sort the final vector based on PR-unsafe reachability (unsafe has priority over safe)
			sort(reachable[ai].begin(), reachable[ai].end(), compare_access_bool_pairs);
		}

		if (verbose) {
			cout << "PRplus_unsafe:" << endl;
			for (auto i : PRplus_unsafe) {
				for (auto j : i.second) {
					cout << "(" << i.first << ", " << j << ")" << endl;
				}
			}
		}


		// Store a set of pairs of accesses that are unsafely related via CMP (ARM only)
		// This is the RFE (external read from) relation
		if (weakmemmodel == ARM) {
			for (int ai = 0; ai < accesses.size(); ai++) {
				Access& a = accesses.get(ai);
				if (a.type == WRITE) {
					vector<int>& S = CMP.get(ai);
					for (int bi : S) {
						Access& b = accesses.get(bi);
						if (b.type == READ) {
							RFE.insert(ai, bi);
						}
					}
				}
			}
		}

		// Compute CMPt
		ThreadAccessRange t;
		for (int ai = 0; ai < accesses.size(); ai++) {
			vector<int>& S = CMP.get(ai);
			int current_thread = -1;
			int bi;
			for (bi = 0; bi < S.size(); bi++) {
				Access& b = accesses.get(S[bi]);
				if (b.tid != current_thread) {
					// Store the current ThreadAccessRange entry, if we were making one, and start making a new one
					if (current_thread != -1) {
						t.accesses_end = bi;
						CMPt.insert(ai, t);
					}
					current_thread = b.tid;
					t.tid = current_thread;
					t.accesses_begin = bi;
				}
			}
			// Finalise the final entry and store it
			if (bi > 0) {
				t.accesses_end = bi;
				CMPt.insert(ai, t);
			}
		}

		// Perform critical cycle detection, based on Tarjan's algorithm for enumerating the elementary circuits in a graph
		int stacksize = accesses.size();
		if (point_stack_max_size < stacksize) {
			stacksize = point_stack_max_size;
		}
		// vector<bool> mark(accesses.size(), false);
		// StaticStack<int> marked_stack(accesses.size());
		StaticStack<StackItem> point_stack(stacksize+1);

		StackItem st_tmp;
		bool g;
		bool initial_ai_PR_explored;
		int unsafe_explored, PR_explored, initial_loc_count;
		vector<bool> visited_threads(thread_ids.size(), false);
		vector<bool> visited_locs(location_ids.size(), false);
		int number_of_locs_visited, number_of_threads_visited;
		// Set to store the PR-paths that require fencing. The Boolean flag indicates whether an Acum-fence is needed (ARM only)
		set<pair<pair<int, int>, bool>> PR_paths_requiring_fences;

		// Count the number of unsafe elements in the program. As soon as everything is marked for fencing, we can stop (early)
		int unsafe_elements_counter = 0;
		for (int i = 0; i < accesses.size(); i++) {
			for (pair<int, bool> p : reachable[i]) {
				if (PRplus_unsafe.are_related(i, p.first)) {
					unsafe_elements_counter++;
				}
				if (PRplus_unsafe.are_related(p.first, i)) {
					unsafe_elements_counter++;
				}
			}
			// if (RFE.contains(i)) {
			// 	auto it = RFE.get(i);
			// 	unsafe_elements_counter += it->second.size();
			// }
		}
		cout << "Number of unsafe elements in AEG: " << unsafe_elements_counter << endl;

		chrono::steady_clock::time_point cycle_begin = chrono::steady_clock::now();

		// Cycle counter
		int cycle_count = 0;

		for (int s = 0; s < accesses.size() && unsafe_elements_counter > 0; s++) {
			if (!accesses.get(s).local) {
				Access& sa = accesses.get(s);
				st_tmp.init(s, 1);
				//print(st_tmp.loc_count);
				point_stack.push(st_tmp);
				//print(point_stack.peek().loc_count);
				// mark[s] = true;
				// marked_stack.push(s);
				initial_ai_PR_explored = false;
				unsafe_explored = 0;
				PR_explored = 0;
				initial_loc_count = 1;
				visited_locs[sa.location] = true;
				visited_threads[sa.tid] = true;
				number_of_locs_visited = 1;
				number_of_threads_visited = 1;
				while (!point_stack.empty() && unsafe_elements_counter > 0) {
					if (verbose) {
						point_stack.print();
						cout << "initial_PR_explored=" << initial_ai_PR_explored << ", unsafe_explored=" << unsafe_explored << ", PR_explored=" << PR_explored << ", initial_loc_count=" << initial_loc_count << endl;
						cout << "visited locations:" << endl;
						for (int i = 0; i < visited_locs.size(); i++) {
							if (visited_locs[i]) {
								cout << i << endl;
							}
						}
						cout << "visited threads:" << endl;
						for (int i = 0; i < visited_threads.size(); i++) {
							if (visited_threads[i]) {
								cout << i << endl;
							}
						}
					}
					StackItem& v_st = point_stack.peek();
					// Check for bound on length point stack
					int w;
					if (point_stack.size() < 6) {
						w = get_next_edge(v_st, s, initial_loc_count, visited_locs, number_of_locs_visited, visited_threads, number_of_threads_visited, initial_ai_PR_explored,
								unsafe_explored, PR_explored, check_atomicity, RFE, reachable, PRplus_unsafe, CMPt, CMP);
					}
					else {
						w = -1;
					}
					// cout << "returned: " << w << endl;
					// if (w > -1) {
					// 	cout << mark[w] << endl;
					// }
					if (w == -1) {
						// Backtrack
						// if (v_st.cycle_found) {
						// 	while (!marked_stack.empty() && marked_stack.peek() != v_st.aid) {
						// 		mark[marked_stack.peek()] = false;
						// 		marked_stack.pop();
						// 	}
						// 	if (!marked_stack.empty()) {
						// 		mark[v_st.aid] = false;
						// 		marked_stack.pop();
						// 	}
						// }
						g = v_st.cycle_found;
						point_stack.pop();
						// Update info relevant for critical cycle conditions
						StackItem& v_st_next = point_stack.peek();
						Access& va = accesses.get(v_st.aid);
						Access& va_next = accesses.get(v_st_next.aid);
						// Number of initial location visits decreases if va accesses the initial location
						if (va.location == sa.location) {
							initial_loc_count--;
						}
						// If not, and if the next access accesses another location, remove the location from the visited locations set
						else if (va.location != va_next.location) {
							visited_locs[va.location] = false;
							number_of_locs_visited--;
						}
						// If we backtrack over a thread, remove it from the visited threads set
						if (va.tid != sa.tid && va.tid != va_next.tid) {
							visited_threads[va.tid] = false;
							number_of_threads_visited--;
						}
						// If we are back at the start, we have not explored a PR-path initially
						if (point_stack.size() == 1) {
							initial_ai_PR_explored = false;
						}
						// Update the number of explored unsafe elements
						if ((v_st_next.edge_type == PREDGE && (PRplus_unsafe.are_related(v_st.aid, v_st_next.aid) || PRplus_unsafe.are_related(v_st_next.aid, v_st.aid)))
								|| (v_st_next.edge_type == CMPEDGE && (RFE.are_related(v_st_next.aid, v_st.aid) || RFE.are_related(v_st.aid, v_st_next.aid)))) {
							unsafe_explored--;
						}
						// Update the number of explored PR-paths
						if (v_st_next.edge_type == PREDGE) {
							PR_explored--;
						}
						if (!point_stack.empty()) {
							v_st_next.cycle_found = v_st_next.cycle_found || g;
						}
					}
					else if (w == s) {
						// Process cycle on point stack. Determine in both directions whether the undirected cycle actually represents a directed cycle.
						bool mark_next_PR = false;
						bool is_directed_cycle = true;
						bool unsafe_element_occurred = false;
						for (vector<StackItem>::iterator st = point_stack.begin(); st != point_stack.end(); ++st) {
							vector<StackItem>::iterator st_next;
							if (st+1 != point_stack.end()) {
								st_next = st+1;
							}
							else {
								st_next = point_stack.begin();
							}
							// If there is actually no PR-path from st->aid to st_next->aid, stop interpreting in this direction of the cycle, unless
							// both accesses stem from the same instruction and we are reasoning about atomicity
							if (st->edge_type == PREDGE) {
								if (!PRplus.are_related(st->aid, st_next->aid) && !(check_atomicity && PRplus_intra_instr.are_related(st_next->aid, st->aid))) {
									is_directed_cycle = false;
									break;
								}
								// Is the PR-path unsafe in the specific direction?
								if (PRplus_unsafe.are_related(st->aid, st_next->aid)) {
									unsafe_element_occurred = true;
								}
							}
							else if (weakmemmodel == ARM) {
								Access& a = accesses.get(st->aid);
								Access& b = accesses.get(st_next->aid);
								if (RFE.are_related(st->aid, st_next->aid)) {
									unsafe_element_occurred = true;
								}
							}
						}
						if (is_directed_cycle && unsafe_element_occurred) {
							cycle_count++;
							if (verbose) {
								cout << "Directed cycle ->!" << endl;
								point_stack.print();
								cout << "initial_PR_explored=" << initial_ai_PR_explored << ", unsafe_explored=" << unsafe_explored << ", PR_explored=" << PR_explored << ", initial_loc_count=" << initial_loc_count << endl;
								cout << "visited locations:" << endl;
								for (int i = 0; i < visited_locs.size(); i++) {
									if (visited_locs[i]) {
										cout << i << endl;
									}
								}
								cout << "visited threads:" << endl;
								for (int i = 0; i < visited_threads.size(); i++) {
									if (visited_threads[i]) {
										cout << i << endl;
									}
								}
							}
							// Scan the stack again, and mark the involved unsafe PR-edges for fencing.
							for (vector<StackItem>::iterator st = point_stack.begin(); st != point_stack.end(); ++st) {
								vector<StackItem>::iterator st_next;
								if (st+1 != point_stack.end()) {
									st_next = st+1;
								}
								else {
									st_next = point_stack.begin();
								}
								if (st->edge_type == PREDGE && (mark_next_PR || PRplus_unsafe.are_related(st->aid, st_next->aid))) {
									PR_paths_requiring_fences.insert(pair<pair<int, int>, bool>(pair<int, int>(st->aid, st_next->aid), mark_next_PR));
									mark_next_PR = false;
									// Mark the path as safe
									if (PRplus_unsafe.are_related(st->aid, st_next->aid)) {
										PRplus_unsafe.unrelate(st->aid, st_next->aid);
										unsafe_elements_counter--;
										if (!PRplus_unsafe.are_related(st_next->aid, st->aid)) {
											unsafe_explored--;
										}
									}
								}
								else if (weakmemmodel == ARM && st->edge_type == CMPEDGE) {
									// If we observe an unsafe CMP edge (in the right direction), mark the subsequent PR-path for fencing
									Access& a = accesses.get(st->aid);
									Access& b = accesses.get(st_next->aid);
									if (RFE.are_related(st->aid, st_next->aid)) {
										mark_next_PR = true;
									}
								}
							}
							// Continue looking for a PR-path to fence, if needed
							if (mark_next_PR) {
								for (vector<StackItem>::iterator st = point_stack.begin(); st != point_stack.end(); ++st) {
									vector<StackItem>::iterator st_next;
									if (st+1 != point_stack.end()) {
										st_next = st+1;
									}
									else {
										st_next = point_stack.begin();
									}
									if (st->edge_type == PREDGE) {
										PR_paths_requiring_fences.insert(pair<pair<int, int>, bool>(pair<int, int>(st->aid, st_next->aid), true));
										break;
									}
								}
							}
						}
						// Consider the other direction
						mark_next_PR = false;
						is_directed_cycle = true;
						unsafe_element_occurred = false;
						// point_stack.print();
						for (vector<StackItem>::reverse_iterator st = point_stack.rbegin(); st != point_stack.rend(); ++st) {
							vector<StackItem>::reverse_iterator st_next;
							if (st+1 != point_stack.rend()) {
								st_next = st+1;
							}
							else {
								st_next = point_stack.rbegin();
							}
							// cout << "from " << st_next->aid << " to " << st->aid << endl;
							// If there is actually no PR-path from st->aid to st_next->aid, stop interpreting in this direction of the cycle, unless
							// both accesses stem from the same instruction and we are reasoning about atomicity
							if (st_next->edge_type == PREDGE) {
								if (!PRplus.are_related(st->aid, st_next->aid) && !(check_atomicity && PRplus_intra_instr.are_related(st_next->aid, st->aid))) {
									// cout << "Not a directed cycle!" << endl;
									// cout << st_next->aid << endl;
									// cout << st->aid << endl;
									is_directed_cycle = false;
									break;
								}
								// Is the PR-path unsafe in the specific direction?
								if (PRplus_unsafe.are_related(st->aid, st_next->aid)) {
									unsafe_element_occurred = true;
								}
							}
							else if (weakmemmodel == ARM) {
								Access& a = accesses.get(st->aid);
								Access& b = accesses.get(st_next->aid);
								if (RFE.are_related(st->aid, st_next->aid)) {
									unsafe_element_occurred = true;
								}
							}
						}
						if (is_directed_cycle && unsafe_element_occurred) {
							cycle_count++;
							if (verbose) {
								cout << "Directed cycle <-!" << endl;
								point_stack.print();
								cout << "initial_PR_explored=" << initial_ai_PR_explored << ", unsafe_explored=" << unsafe_explored << ", PR_explored=" << PR_explored << ", initial_loc_count=" << initial_loc_count << endl;
								cout << "visited locations:" << endl;
								for (int i = 0; i < visited_locs.size(); i++) {
									if (visited_locs[i]) {
										cout << i << endl;
									}
								}
								cout << "visited threads:" << endl;
								for (int i = 0; i < visited_threads.size(); i++) {
									if (visited_threads[i]) {
										cout << i << endl;
									}
								}
							}
							// Scan the stack again, and mark the involved unsafe edges for fencing.
							for (vector<StackItem>::reverse_iterator st = point_stack.rbegin(); st != point_stack.rend(); ++st) {
								vector<StackItem>::reverse_iterator st_next;
								if (st+1 != point_stack.rend()) {
									st_next = st+1;
								}
								else {
									st_next = point_stack.rbegin();
								}
								if (st_next->edge_type == PREDGE && (mark_next_PR || PRplus_unsafe.are_related(st->aid, st_next->aid))) {
									PR_paths_requiring_fences.insert(pair<pair<int, int>, bool>(pair<int, int>(st->aid, st_next->aid), mark_next_PR));
									mark_next_PR = false;
									// Mark the path as safe
									if (PRplus_unsafe.are_related(st->aid, st_next->aid)) {
										PRplus_unsafe.unrelate(st->aid, st_next->aid);
										unsafe_elements_counter--;
										if (!PRplus_unsafe.are_related(st_next->aid, st->aid)) {
											unsafe_explored--;
										}
									}
								}
								else if (weakmemmodel == ARM && st_next->edge_type == CMPEDGE) {
									// If we observe an unsafe CMP edge (in the right direction), mark the subsequent PR-path for fencing
									Access& a = accesses.get(st->aid);
									Access& b = accesses.get(st_next->aid);
									if (RFE.are_related(st->aid, st_next->aid)) {
										mark_next_PR = true;
									}
								}
							}
							// Continue looking for a PR-path to fence, if needed
							if (mark_next_PR) {
								for (vector<StackItem>::reverse_iterator st = point_stack.rbegin(); st != point_stack.rend(); ++st) {
									vector<StackItem>::reverse_iterator st_next;
									if (st+1 != point_stack.rend()) {
										st_next = st+1;
									}
									else {
										st_next = point_stack.rbegin();
									}
									if (st->edge_type == PREDGE) {
										PR_paths_requiring_fences.insert(pair<pair<int, int>, bool>(pair<int, int>(st->aid, st_next->aid), true));
										break;
									}
								}
							}
						}
						//cout << "Cycle!" << endl;
						v_st.cycle_found = true;
					}
					// else if (!mark[w]) {
					else {
						Access& va = accesses.get(v_st.aid);
						Access& wa = accesses.get(w);
						if (v_st.edge_type == PREDGE) {
							PR_explored++;
							if (PRplus_unsafe.are_related(v_st.aid, w) || PRplus_unsafe.are_related(w, v_st.aid)) {
								unsafe_explored++;
							}
							if (v_st.aid == s) {
								initial_ai_PR_explored = true;
							}
						}
						else {
							if (!visited_threads[wa.tid]) {
								visited_threads[wa.tid] = true;
								number_of_threads_visited++;
							}
							if (RFE.are_related(v_st.aid, w) || RFE.are_related(w, v_st.aid)) {
								unsafe_explored++;
							}
							if (v_st.aid == s) {
								initial_ai_PR_explored = false;
							}
						}
						if (!visited_locs[wa.location]) {
							visited_locs[wa.location] = true;
							number_of_locs_visited++;
						}
						int loc_count = 1;
						if (wa.location == sa.location) {
							loc_count = initial_loc_count+1;
							initial_loc_count++;
						}
						else if (va.location == wa.location) {
							loc_count = v_st.loc_count+1;
						}
						if (va.tid == wa.tid) {
							st_tmp.init_CMP(w, loc_count);
						}
						else {
							st_tmp.init(w, loc_count);
						}
						point_stack.push(st_tmp);
						// mark[w] = true;
						// marked_stack.push(w);
						continue;
					}
				}
				// while (!marked_stack.empty()) {
				// 	mark[marked_stack.peek()] = false;
				// 	marked_stack.pop();
				// }
				visited_locs[sa.location] = false;
				visited_threads[sa.tid] = false;
			}
		}
		// cout << "Marked for fencing:" << endl;
		// for (auto p : PR_paths_requiring_fences) {
		// 	cout << p.first << " -PR-> " << p.second << endl;
		// 	cout << p.first << " is part of static instruction " << accesses.get(p.first).ipos << endl;
		// 	cout << p.second << " is part of static instruction " << accesses.get(p.second).ipos << endl;
		// 	cout << p.first << " is part of dynamic instructions ";
		// 	for (int i : accesses.get(p.first).instr) {
		// 		Instruction& instr = instructions.find(i)->second;
		// 		if (instr.accesses.find(p.first) != instr.accesses.end()) {
		// 			cout << " " << i;
		// 		}
		// 	}
		// 	cout << endl;
		// 	cout << p.second << " is part of dynamic instructions ";
		// 	for (int i : accesses.get(p.second).instr) {
		// 		Instruction& instr = instructions.find(i)->second;
		// 		if (instr.accesses.find(p.second) != instr.accesses.end()) {
		// 			cout << " " << i;
		// 		}
		// 	}
		// 	cout << endl;
		// }

		chrono::steady_clock::time_point cycle_end = chrono::steady_clock::now();

		// Postprocess the marked PR-pairs, to identify how many fences should be placed, of which kind, and how many
		StaticStack<StackItem_fencing> processing_stack(instructions.size());
		// set to store for each instruction after which a fence must be placed the set of instructions where such a fence can be placed
		set<fence_candidate_t> fence_candidates;

		map<int, set<int>> closed_instr;
		vector<bool> marked_instr(instructions.size(), false);
		// Set of instructions that have fences placed inside/after them
		set<int> fenced_instructions;
		// Set of instructions that have A_cumulative fences placed inside/after them (see Don't Sit, only relevant for ARM)
		set<int> Acum_fenced_instructions;
		// Set of instructions that are locked (atomicity checking only)
		set<int> locked_instructions;
		// Temporary set of fences
		set<int> new_fences;

		StackItem_fencing st_f;
		set<int> instr2_ids;
		set<pair<int, bool>> candidateset;
		fence_candidate_t fcand;
		for (pair<pair<int, int>, bool> p : PR_paths_requiring_fences) {
			int pfirst = p.first.first;
			int psecond = p.first.second;
			bool Acum_needed = p.second;
			// cout << "PR_path under consideration: " << pfirst << " " << psecond << " " << Acum_needed << endl;
			vector<int>& instr1_ids = accesses.get(pfirst).instr;
			instr2_ids.clear();
			instr2_ids.insert(accesses.get(psecond).instr.begin(), accesses.get(psecond).instr.end());
			if (accesses.get(pfirst).ipos == accesses.get(psecond).ipos) {
				if (check_atomicity && PRplus_intra_instr.are_related(psecond, pfirst)) {
					locked_instructions.insert(accesses.get(pfirst).ipos);
				}
				else {
					candidateset.clear();
					// Add the goal instruction
					candidateset.insert(pair<int, bool>(accesses.get(psecond).ipos, true));
					fcand.start = accesses.get(pfirst).ipos;
					fcand.Acum_needed = Acum_needed;
					fcand.instr = candidateset;
					fence_candidates.insert(fcand);
				}
				// cout << "adding fence to instruction " << accesses.get(p.first).ipos << endl;
			}
			else {
				vector<int>& instr1_ids = accesses.get(pfirst).instr;
				instr2_ids.clear();
				instr2_ids.insert(accesses.get(psecond).instr.begin(), accesses.get(psecond).instr.end());
				// cout << "instr2_ids: ";
				// for (int i : instr2_ids) {
				// 	cout << " " << i;
				// }
				// cout << endl;
				for (int instr1_id : instr1_ids) {
					// Put instr1_id on the processing stack, and search for a path leading to an instruction in instr2_ids
					closed_instr.clear();
					st_f.instruction = instr1_id;
					st_f.next.clear();
					if (PRinstr.contains(instr1_id)) {
						st_f.next.insert(st_f.next.end(), PRinstr.get(instr1_id)->second.begin(), PRinstr.get(instr1_id)->second.end());
					}
					st_f.essential_instr.clear();
					processing_stack.push(st_f);
					marked_instr[instr1_id] = true;
					bool progress = true;
					while (!processing_stack.empty()) {
						StackItem_fencing& st_top = processing_stack.peek();
						// cout << "checking for " << st_top.instruction << endl;
						//int ipos = instructions.find(st_top.instruction)->second.pos;
						int ipos = st_top.instruction;
						progress = false;
						while (!st_top.next.empty()) {
							int next_instr_id = st_top.next.back();
							// cout << "next out: " << next_instr_id << endl;
							st_top.next.pop_back();
							if (!set_contains(instr2_ids, next_instr_id)) {
								auto it = closed_instr.find(next_instr_id);
								if (it != closed_instr.end()) {
									// Instruction already visited. Update the set of essential instructions
									if (st_top.essential_instr.empty()) {
										st_top.essential_instr.insert(it->second.begin(), it->second.end());
										// cout << "adding essentials:" << endl;
										// for (int i : st_top.essential_instr) {
										// 	cout << " " << i << endl;
										// }
									}
									else {
										for (auto sit = st_top.essential_instr.begin(); sit != st_top.essential_instr.end();) {
											if (*sit != ipos && it->second.find(*sit) == it->second.end()) {
												sit = st_top.essential_instr.erase(sit);
											}
											else {
												++sit;
											}
										}
										// cout << "new essentials:" << endl;
										// for (int i : st_top.essential_instr) {
										// 	cout << " " << i << endl;
										// }
									}
								}
								else if (!marked_instr[next_instr_id]) {
									st_f.instruction = next_instr_id;
									// cout << "continuing with " << next_instr_id << endl;
									st_f.next.clear();
									if (PRinstr.contains(next_instr_id)) {
										st_f.next.insert(st_f.next.end(), PRinstr.get(next_instr_id)->second.begin(), PRinstr.get(next_instr_id)->second.end());
									}
									st_f.essential_instr.clear();
									processing_stack.push(st_f);
									marked_instr[next_instr_id] = true;
									progress = true;
									break;
								}
							}
							// else {
							// 	// We reached one of the instructions in instr2_ids. Add instr1_id as an essential instruction to reach it
							// 	//cout << "goal instruction found, adding " << ipos << endl;
							// 	st_top.essential_instr.insert(ipos);
							// }
						}
						if (!progress) {
							// cout << "closing " << st_top.instruction << ": " << endl;
							// st_top.essential_instr.insert(ipos);
							// for (int i : st_top.essential_instr) {
							// 	cout << i << " ";
							// }
							// cout << endl;
							st_top.essential_instr.insert(ipos);
							closed_instr.insert(pair<int, set<int>>(st_top.instruction, st_top.essential_instr));
							processing_stack.pop();
							marked_instr[st_top.instruction] = false;
							if (!processing_stack.empty()) {
								StackItem_fencing& st_prev = processing_stack.peek();
								if (st_prev.essential_instr.empty()) {
									st_prev.essential_instr.insert(st_top.essential_instr.begin(), st_top.essential_instr.end());
								}
								else {
									for (auto sit = st_prev.essential_instr.begin(); sit != st_prev.essential_instr.end();) {
										if (!set_contains(st_top.essential_instr, *sit)) {
											sit = st_prev.essential_instr.erase(sit);
										}
										else {
											++sit;
										}
									}
								}
							}
						}
					}
					candidateset.clear();
					// Prepare set of candidate pairs
					set<int>& C = closed_instr.find(instr1_id)->second;
					for (int c : C) {
						candidateset.insert(pair<int, bool>(c, true));
					}
					// Add the goal instruction
					candidateset.insert(pair<int, bool>(accesses.get(psecond).ipos, false));
					//auto it = fence_candidates.find(instr1_id);
					//if (it == fence_candidates.end()) {
					//fence_candidates.insert(pair<int, set<pair<int, bool>>>(instr1_id, candidateset));
					fcand.start = instr1_id;
					fcand.Acum_needed = Acum_needed;
					fcand.instr = candidateset;
					fence_candidates.insert(fcand);
					//}
					//else {
					//	it->second.insert(candidateset.begin(), candidateset.end());
					//}
				}
			}
		}
		
		// Place the fences
		vector<int> instr_counts(instructions.size(), 0);
		for (fence_candidate_t can : fence_candidates) {
			for (pair<int, bool> c : can.instr) {
				// if (c.second) {
					instr_counts[c.first]++;
					// cout << "incrementing counter for " << c.first << endl;
				// }
			}
		}
		
		if (verbose) {
			cout << "fence candidates:" << endl;
			for (fence_candidate_t can : fence_candidates) {
				cout << can.start << ": ";
				for (pair<int, bool> c : can.instr) {
					cout << " (" << c.first << ", " << c.second << ")";
				}
				cout << endl;
			}
		}
		// Atomicity checking only: lock the instructions marked for locking
		if (check_atomicity) {
			for (int i : locked_instructions) {
				//cout << "locking " << i << endl;
				for (auto sit = fence_candidates.begin(); sit != fence_candidates.end();) {
					if (set_contains(sit->instr, (pair<int, bool>(i, true))) || set_contains(sit->instr, (pair<int, bool>(i, false)))) {
						for (pair<int, bool> c : sit->instr) {
							// if (c.second) {
								instr_counts[c.first]--;
							// }
						}
						sit = fence_candidates.erase(sit);
					}
					else {
						++sit;
					}
				}
			}
		}
		// Select instructions that are single candidates for at least one fencing requirement
		for (fence_candidate_t can : fence_candidates) {
			if (can.instr.size() == 1) {
				new_fences.insert((can.instr.begin())->first);
			}
		}
		while (!fence_candidates.empty()) {
			for (int i : new_fences) {
				bool needs_Acum_fence = false;
				// cout << "fencing " << i << endl;
				for (auto sit = fence_candidates.begin(); sit != fence_candidates.end();) {
					if (set_contains(sit->instr, (pair<int, bool>(i, true))) || set_contains(sit->instr, (pair<int, bool>(i, false)))) {
						for (pair<int, bool> c : sit->instr) {
							//if (c.second) {
								// cout << "decrementing counter of " << c.first << endl;
								// cout << instr_counts[c.first] << endl;
								instr_counts[c.first]--;
								// cout << instr_counts[c.first] << endl;
							//}
						}
						if (sit->Acum_needed) {
							needs_Acum_fence = true;
						}
						sit = fence_candidates.erase(sit);
					}
					else {
						++sit;
					}
				}
				if (needs_Acum_fence) {
					Acum_fenced_instructions.insert(i);
				}
				else {
					fenced_instructions.insert(i);
				}
			}
			if (verbose) {
				cout << "fence candidates:" << endl;
				for (fence_candidate_t can : fence_candidates) {
					cout << can.start << ": ";
					for (pair<int, bool> c : can.instr) {
						cout << " (" << c.first << ", " << c.second << ")";
					}
					cout << endl;
				}
			}
			new_fences.clear();
			// Find the maximum count in instr_counts
			int max = 0;
			int maxpos = 0;
			for (int i = 0; i < instr_counts.size(); i++) {
				// cout << instr_counts[i] << endl;
				if (instr_counts[i] > max) {
					max = instr_counts[i];
					maxpos = i;
				}
			}
			// If the highest count is larger than 0, mark the instruction for fencing, and repeat
			if (max > 0) {
				// cout << "here" << endl;
				new_fences.insert(maxpos);
			}
		}

		chrono::steady_clock::time_point end = chrono::steady_clock::now();

		cout << "Number of directed cycles found: " << cycle_count << endl;
		if (check_atomicity) {
			cout << "Locks placed on " << locked_instructions.size() << " instructions!" << endl;
			if (locked_instructions.size() > 0) {
				cout << "The following instructions are locked: ";
				for (int i : locked_instructions) {
					cout << i << " ";
				}
				cout << endl;
			}
		}
		cout << "Fences placed in/after " << fenced_instructions.size() << " instructions!" << endl;
		if (fenced_instructions.size() > 0) {
			cout << "The following instructions are fenced: ";
			for (int i : fenced_instructions) {
				cout << i << " ";
			}
			cout << endl;
		}
		if (weakmemmodel == ARM) {
			cout << "A_Cumulative fences placed in/after " << Acum_fenced_instructions.size() << " instructions!" << endl;
			if (Acum_fenced_instructions.size() > 0) {
				cout << "The following instructions are A_Cumulatively fenced: ";
				for (int i : Acum_fenced_instructions) {
					cout << i << " ";
				}
				cout << endl;
			}
		}

		cout << "Cycle detection time = " << chrono::duration_cast<chrono::microseconds>(cycle_end - cycle_begin).count() << "[µs]" << endl;
		cout << "Total analysis time = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]" << endl;

		// cout << "PR:" << endl;
		// for (auto i : PR) {
		// 	for (auto j : i.second) {
		// 		cout << "(" << i.first << ", " << j << ")" << endl;
		// 	}
		// }
		// cout << "The PPRs: " << endl;
		// for (int i = 0; i < instructions.size(); i++) {
		// 	cout << "For instruction " << i << ":" << endl;
		// 	cout << "Accesses of the instruction: ";
		// 	Instruction& instr = instructions.find(i)->second;
		// 	for (int n : instr.accesses) {
		// 		cout << n << " ";
		// 	}
		// 	cout << endl;
		// 	for (auto j : PPR[i]) {
		// 		for (auto k : j.second) {
		// 			cout << "(" << j.first << ", " << k << ")" << endl;
		// 		}
		// 	}
		// }
		// cout << "PRplus:" << endl;
		// for (auto i : PRplus) {
		// 	for (auto j : i.second) {
		// 		cout << "(" << i.first << ", " << j << ")" << endl;
		// 	}
		// }
		// cout << "PRplus_intra_instr:" << endl;
		// for (auto i : PRplus_intra_instr) {
		// 	for (auto j : i.second) {
		// 		cout << "(" << i.first << ", " << j << ")" << endl;
		// 	}
		// }
		// cout << "PRplus_unsafe:" << endl;
		// for (auto i : PRplus_unsafe) {
		// 	for (auto j : i.second) {
		// 		cout << "(" << i.first << ", " << j << ")" << endl;
		// 	}
		// }
		// cout << "DP:" << endl;
		// for (auto i : DP) {
		// 	for (auto j : i.second) {
		// 		cout << "(" << i.first << ", " << j << ")" << endl;
		// 	}
		// }
		// cout << "CTRL:" << endl;
		// for (auto i : CTRL) {
		// 	for (auto j : i.second) {
		// 		cout << "(" << i.first << ", " << j << ")" << endl;
		// 	}
		// }
		// cout << "PRinstr:" << endl;
		// for (auto i : PRinstr) {
		// 	for (auto j : i.second) {
		// 		cout << "(" << i.first << ", " << j << ")" << endl;
		// 	}
		// }
		// cout << "CMP:" << endl;
		// for (int i = 0; i < CMP.size(); i++) {
		// 	vector<int>& S = CMP.get(i);
		// 	for (auto j : S) {
		// 		cout << "(" << i << ", " << j << ")" << endl;
		// 	}
		// }
		// cout << "CMPt:" << endl;
		// for (int i = 0; i < CMPt.size(); i++) {
		// 	vector<ThreadAccessRange>& S = CMPt.get(i);
		// 	for (ThreadAccessRange j : S) {
		// 		cout << "(" << i << ", " << j.tid << ", " << j.accesses_begin << ", " << j.accesses_end << ")" << endl;
		// 	}
		// }	
	}
	else {
		cout << "File does not exist!" << endl;
		exit(-1);
	}
}
