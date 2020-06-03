//
//  seqcon-analyser.cpp
//  
//
//  Created by Anton Wijs on 11/05/2020.
//

#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <set>
#include <algorithm>

using namespace std;

// Memory model
enum MM { TSO, PSO, ARM };
// Access type
enum AccessType { READ, WRITE };
// Type of edge in an Abstract Event Graph
enum edgeType { PRUNSAFE, PRSAFE, CMPEDGE };

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
				storage.insert(storage.end(), item);
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
				return it->second.find(j) != it->second.end();
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
	int instruction; // Instruction of which the access is part
	int tid; // Thread executing the access
};

// < operator for Accesses, to be able to store them in a map
bool operator<(const Access& a1, const Access& a2) {
     return (a1.location<a2.location || (a1.location==a2.location && a1.local<a2.local)
     	|| (a1.location==a2.location && a1.local==a2.local && a1.type < a2.type)
     	|| (a1.location==a2.location && a1.local==a2.local && a1.type==a2.type && a1.instruction < a2.instruction)
     	|| (a1.location==a2.location && a1.local==a2.local && a1.type==a2.type && a1.instruction==a2.instruction && a1.tid < a2.tid));
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
	int outgoing_begin = -1; // Start of list of outgoing transitions
	int outgoing_end = -1; // End of list of outgoing transitions
};

// Struct to store LTS transition
struct Transition {
	int target; // Target state
	map<int, Instruction>::iterator instruction; // Instruction associated with the transition
};

// Struct to store CMP lookup info (for an access, thread pair)
struct ThreadAccessRange {
	int tid;
	int accesses_begin;
	int accesses_end;
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
		this->edge_type = PRUNSAFE;
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
		if (edge_type == PRUNSAFE) {
			cout << "PR UNSAFE";
		}
		else if (edge_type == PRSAFE) {
			cout << "PR SAFE";
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
			return stack.begin() + top;
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

// Comparison function for sorting (access ID, bool) pairs
bool compare_access_bool_pairs(pair<int, bool> p1, pair<int, bool> p2) {
	
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
				vector<BiRelation>& PPR, BiRelation PR, Relation DP, Relation CTRL, bool from_outside_instr) {
	// Get the PPR-predecessors of ai.
	bool moved_into = false;
	auto instr = (instructions.find(instr_id));
	set<int> reordered;
	//cout << "reordering " << ai << endl;
	if ((!from_outside_instr && PR.contains_rev(ai)) || (from_outside_instr && !(instr->second.top_accs.empty()))) {
		set<int> set1, set2;
		set<int>& open = set1;
		set<int>& tmp = set1;
		set<int>& next = set2;

		if (!from_outside_instr) {
			auto it_pred = PR.get_rev(ai);
			for (auto bi : it_pred->second) {
				if (instr->second.accesses.find(bi) != instr->second.accesses.end()) {
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
				bool check = true;
				if (PPR[instr_id].contains(bi)) {
					auto it = PPR[instr_id].get(bi);
					for (int ci : it->second) {
						if (reordered.find(ci) == reordered.end()) {
							check = false;
						}
					}
				}
				if (check) {
					if (can_reorder(bi, ai, mmodel, DP, CTRL)) {
						reordered.insert(bi);
						if (!from_outside_instr) {
							if (PR.contains_rev(bi)) {
								auto it = PR.get_rev(bi);
								for (auto ci : it->second) {
									if (instr->second.accesses.find(ci) != instr->second.accesses.end()) {
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
							if (PR.contains_rev(bi)) {
								map<int, set<int>>::iterator it_bi_pr_pred = PR.get_rev(bi);
								map<int, set<int>>::iterator it_bi_ppr_pred;
								bool has_PPR_preds = false;
								if (PPR[instr_id].contains_rev(bi)) {
									it_bi_ppr_pred = PPR[instr_id].get_rev(bi);
									has_PPR_preds = true;
								}
								for (int ci : it_bi_pr_pred->second) {
									if (instr->second.accesses.find(ci) != instr->second.accesses.end()) {
										if (!has_PPR_preds || it_bi_ppr_pred->second.find(ci) == it_bi_ppr_pred->second.end()) {
											next.insert(ci);
										}
									}
								}
							}
						}
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
		if (instr->second.accesses.find(ai) == instr->second.accesses.end()) {
			moved_into = true;
			instr->second.accesses.insert(ai);
		}
	}
	// If ai ends up not having PPR-predecessors, it is a new bottom access
	if (!PPR[instr_id].contains_rev(ai) && (from_outside_instr || PR.contains_rev(ai))) {
		instr->second.bottom_accs.insert(instr->second.bottom_accs.end(), ai);
	}
	// If ai ends up not having PPR-successors, it is a new top access
	if (!PPR[instr_id].contains(ai) && (from_outside_instr || PR.contains(ai))) {
		instr->second.top_accs.insert(instr->second.top_accs.end(), ai);
		// PPR-predecessors are not top accesses
		if (PPR[instr_id].contains_rev(ai)) {
			auto it = PPR[instr_id].get_rev(ai);
			for (int bi : it->second) {
				instr->second.top_accs.erase(bi);
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
int get_next_edge(StackItem& s, int initial_ai, int& initial_loc_count, set<int>& visited_locs, set<int>& visited_threads,
						bool& initial_ai_PR_explored, int& unsafe_explored, int& PR_explored, bool atomicity_check,
						Relation RFE, vector<vector<int>> PRsafe, vector<vector<int>> PRunsafe,
						VectorRelation<ThreadAccessRange> CMPt, VectorRelation<int> CMP) {
	int result = -1;
	int selected;
	int initial_tid = accesses.get(initial_ai).tid;
	// Consider an unsafe PR-path
	if (s.edge_type == PRUNSAFE) {
		// Special case: if we are constructing a cycle visiting just one location, with initially a PR-path, then we are not allowed to select a PR-path.
		// In other words, we have constructed a path a -PR-> b -CMP-> c so far, a, b, c all accessing the same location, and only c -CMP-> a can be selected
		// (not relevant when atomicity checking is not done, as cycles with one location are not interesting).
		Access& a = accesses.get(s.aid);
		Access& a_initial = accesses.get(initial_ai);
		if (!(a.location == a_initial.location && initial_ai_PR_explored && s.loc_count == 3)) {
			for (int i = s.edge_index+1; i < PRunsafe[s.aid].size(); i++) {
				selected = PRunsafe[s.aid][i];
				if (selected >= initial_ai) {
					Access& b = accesses.get(selected);
					// Either we are considering a thread other than the first one, or we have only selected one thread so far (a cycle cannot be closed yet),
					// or we must close a cycle. In that case, this step should lead to the initially selected access.
					if (a.tid != initial_tid || visited_threads.size() == 1 || selected == initial_ai) {
						// Either we are atomicity checking, or, if not, the locations of the previous and next access are not the same (see Don't Sit On The Fence)
						if (atomicity_check || a.location != b.location) {
							// Cycle closing location condition: if we are revisiting the initial location, then we must keep selecting that location to close the cycle.
							// If not, then we may either select a.location again, if s.loc_count allows it, or we can select a previously unvisited location.
							if ( (a.location == a_initial.location && visited_locs.size() > 1 && s.loc_count < 3 && b.location == a_initial.location) ||
								 ( !(a.location == a_initial.location && visited_locs.size() > 1) &&
								 		((s.loc_count < 3 && a.location == b.location) || visited_locs.find(b.location) == visited_locs.end())
								 			|| (initial_loc_count < 3 && b.location == a_initial.location) )
								) {
								s.edge_index = i;
								result = selected;
								PR_explored++;
								unsafe_explored++;
								if (s.aid == initial_ai) {
									initial_ai_PR_explored = true;
								}
								visited_locs.insert(b.location);
								break;
							}
						}
					}
				}
			}
		}
		if (result == -1) {
			s.edge_type = PRSAFE;
			s.edge_index = -1;
			s.t_index = 0;
		}
	}
	// Consider a safe PR-path
	if (s.edge_type == PRSAFE) {
		// Special case: if we are constructing a cycle visiting just one location, with initially a PR-path, then we are not allowed to select a PR-path.
		// In other words, we have constructed a path a -PR-> b -CMP-> c so far, a, b, c all accessing the same location, and only c -CMP-> a can be selected
		// (not relevant when atomicity checking is not done, as cycles with one location are not interesting).
		Access& a = accesses.get(s.aid);
		Access& a_initial = accesses.get(initial_ai);
		if (!(a.location == a_initial.location && initial_ai_PR_explored && s.loc_count == 3)) {
			for (int i = s.edge_index+1; i < PRsafe[s.aid].size(); i++) {
				selected = PRsafe[s.aid][i];
				cout << "checking " << selected << endl;
				if (selected >= initial_ai) {
					Access& b = accesses.get(selected);
					// Either we are considering a thread other than the first one, or we have only selected one thread so far (a cycle cannot be closed yet),
					// or we must close a cycle. In that case, this step should lead to the initially selected access, and we must have at least one unsafe element
					// in the cycle.
					if (a.tid != initial_tid || visited_threads.size() == 1 || (selected == initial_ai && unsafe_explored > 0)) {
						// Either we are atomicity checking, or, if not, the locations of the previous and next access are not the same (see Don't Sit On The Fence)
						if (atomicity_check || a.location != b.location) {
							// Cycle closing location condition: if we are revisiting the initial location, then we must keep selecting that location to close the cycle.
							// If not, then we may either select a.location again, if s.loc_count allows it, or we can select a previously unvisited location.
							if ( (a.location == a_initial.location && visited_locs.size() > 1 && s.loc_count < 3 && b.location == a_initial.location) ||
								 ( !(a.location == a_initial.location && visited_locs.size() > 1) &&
								 		((s.loc_count < 3 && a.location == b.location) || visited_locs.find(b.location) == visited_locs.end()
								 			|| (initial_loc_count < 3 && b.location == a_initial.location)) )
								) {
								s.edge_index = i;
								result = selected;
								PR_explored++;
								if (s.aid == initial_ai) {
									initial_ai_PR_explored = true;
								}
								visited_locs.insert(b.location);
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
		}
	}
	// Consider a CMP-edge
	if (s.edge_type == CMPEDGE) {
		vector<ThreadAccessRange>& out = CMPt.get(s.aid);
		for (int i = s.t_index; i < out.size(); i++) {
			if (visited_threads.find(out[i].tid) == visited_threads.end() || out[i].tid == initial_tid) {
				for (int j = (i == s.t_index ? s.edge_index+1 : out[i].accesses_begin); j < out[i].accesses_end; j++) {
					selected = CMP.get_element(s.aid, j);
					if (selected >= initial_ai) {
						cout << "CMP: checking " << selected << endl;
						Access& a = accesses.get(s.aid);
						Access& b = accesses.get(selected);
						Access& a_initial = accesses.get(initial_ai);
						// If we return to the initial thread, then we can either select an access different from the initial one (if we did not at the
						// start explore a PR-path of that thread), or we can select the initial access, by which we close a cycle.
						// In that case, the cycle should be critical, i.e., a PR-path must have been explored, and either an unsafe element must have been explored,
						// or the selected CMP-edge is unsafe.
						if (out[i].tid != initial_tid || (selected != initial_ai && !initial_ai_PR_explored) ||
								(selected == initial_ai && PR_explored > 0 && (unsafe_explored > 0 || RFE.are_related(s.aid, j)))) {
							// Cycle closing location condition: either we must select initial_ai (special case), or, if we are revisiting the initial location,
							// then we must keep selecting that location to close the cycle. If not, then we may either select a.location again, if s.loc_count allows it,
							// or we can select a previously unvisited location.
							if (a.location == a_initial.location && initial_ai_PR_explored && s.loc_count == 3 && selected != initial_ai) {
								continue;
							}
							if ( (a.location == a_initial.location && visited_locs.size() > 1 && s.loc_count < 3 && b.location == a_initial.location) ||
								 ( !(a.location == a_initial.location && visited_locs.size() > 1) &&
								 		((s.loc_count < 3 && a.location == b.location) || visited_locs.find(b.location) == visited_locs.end())
								 			|| (initial_loc_count < 3 && b.location == a_initial.location) )
								) {
								s.edge_index = j;
								s.t_index = i;
								result = selected;
								visited_threads.insert(out[i].tid);
								if (RFE.are_related(s.aid, j)) {
									unsafe_explored = true;
								}
								if (s.aid == initial_ai) {
									initial_ai_PR_explored = false;
								}
								visited_locs.insert(b.location);								
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

int main (int argc, char *argv[]) {
	string modelname = "";
	MM weakmemmodel = TSO;
	bool check_atomicity = false;

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
		}
		else if (strcmp(argv[i], "-h") == 0) {
			cout << "Usage: seqcon-analyser [-wsa] model" << endl;
			cout << "" << endl;
			cout << "Check for sequentially inconsistent behaviour in the given .aut file containing the state space of an SLCO model, unless '-s' is used." << endl;
			cout << " -w                                  weak memory model to consider (TSO,PSO,ARM)  (default: TSO)" << endl;
			cout << " -s                                  apply only static analysis (ignore state space) (default: no)" << endl;
			cout << "                                       -> this option requires a .instr file listing the instructions of the SLCO model" << endl;
			cout << " -a                                  apply atomicity checking in combination with SC checking (default: no)" << endl;
		}
		else {
			modelname = string(argv[i]);
		}
	}

	// PR relation
	BiRelation PR;
	// Transitive closure of PR
	Relation PRplus;
	// PR relation at instruction level
	BiRelation PRinstr;
	// Dependency relation DP
	// Covers value and address dependencies, and RF for thread-local variables
	Relation DP;
	// CTRL dependency relation
	Relation CTRL;
	// Unsafe CMP relation, corresponding for ARM with RFE (external read from)
	Relation RFE;
	// Unsafe PR selfloops
	set<int> PRunsafe_selfloops;
	
	// Various IndexedMaps to keep track of information extracted from the LTS
	IndexedMap<string> instruction_positions; // Maps short instruction position strings to integer IDs
	IndexedMap<string> instruction_ids; // Maps long instruction strings to integer IDs
	map<int, Instruction> instructions;
	IndexedMap<string> thread_ids;
	IndexedMap<string> location_ids;

	// Reading the input LTS
	string line;
	ifstream ltsfile (modelname);
	if (ltsfile.is_open()) {
		// Header: extract number of states
		getline(ltsfile, line);
		size_t sep1 = line.find_last_of(",");
		int nr_of_states = stoi(line.substr(sep1+1, line.find_last_of(")")-(sep1+1)));
		// Create vector for states
		vector<State> lts_states(nr_of_states);
		size_t sep2 = line.find_first_of(",");
		int nr_of_trans = stoi(line.substr(sep2+1, sep1-(sep2+1)));
		// Create vector for transitions
		vector<Transition> lts_transitions(nr_of_trans);

		int current_state_index = -1;
		int current_trans_index = 0;
		// Instruction instance to be used to create new instructions
		Instruction instr;
		// Access instance to be used to create new accesses
		Access acc;
		int prev_src = -1;
		bool first_trans;
		while (getline(ltsfile, line)) {
			// Extract info from transition label, and store the transition
			sep1 = line.find_first_of(",");
			int src = stoi(line.substr(1,sep1-1));
			if (src != prev_src) {
				if (prev_src != -1) {
					lts_states[prev_src].outgoing_end = current_trans_index;
				}
				first_trans = true;
				prev_src = src;
				current_state_index++;
			}
			else {
				first_trans = false;
			}
			sep2 = line.find_first_of("\"", sep1+2);
			string label = line.substr(sep1+2, sep2-(sep1+2));
			int tgt = stoi(line.substr(sep2+2, line.length()-(sep2+2)-1));
			// List of previous and current accesses, used to build intra-instruction PR-relation
			vector<int> prev_accesses, curr_accesses_bottom, curr_accesses_top;

			// Store long instruction string description
			int iid = instruction_ids.find(label);
			// Check if instruction is already stored. If not, create it
			if (iid == -1) {
				iid = instruction_ids.insert(label);
				cout << iid << ": " << label << endl;
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
					cout << statement << endl;
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
							//cout << read << endl;

							// Create and store read access
							int loc = location_ids.insert(read);
							acc.location = loc;
							// cout << loc << endl;
							// Is the variable thread-local? (encoded in name by the fact that ' occurs more than once)
							acc.local = count(read.begin(), read.end(), '\'') > 1;
							// cout << acc.local << endl;
							acc.type = READ;
							// cout << acc.type << endl;
							acc.instruction = ipos;
							// cout << acc.instruction << endl;
							acc.tid = tid;
							// cout << tid << endl;
							int aid = accesses.insert(acc);
							cout << "read " << aid << " : " << read << ": " << label << endl;
							instr.accesses.insert(instr.accesses.end(), aid);
							reads_stored = true;

							if (reads[sep2-1] != 'p') {
								curr_accesses_bottom.insert(curr_accesses_bottom.end(), aid);
							}
							else {
								curr_accesses_top.insert(curr_accesses_bottom.end(), aid);
							}

							if (reads[sep2-1] == 'p') {
								// we have a tuple with a read and a list of address dependencies of that read
								// record the dependencies
								size_t sep3 = sep1+3;
								while (true) {
									size_t sep4 = reads.find_first_of(",]", sep3);
									string depread = reads.substr(sep3, sep4-sep3);
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

							// Create and store write access
							int loc = location_ids.insert(write);
							acc.location = loc;
							// cout << loc << endl;
							// Is the variable thread-local? (encoded in name by the fact that ' occurs more than once)
							acc.local = count(write.begin(), write.end(), '\'') > 1;
							// cout << acc.local << endl;
							acc.type = WRITE;
							// cout << acc.type << endl;
							acc.instruction = ipos;
							// cout << acc.instruction << endl;
							acc.tid = tid;
							// cout << tid << endl;
							int aid = accesses.insert(acc);
							cout << "write " << aid << " : " << write << ": " << label << endl;
							instr.accesses.insert(instr.accesses.end(), aid);
							curr_accesses_bottom.insert(curr_accesses_bottom.end(), aid);

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
			// Get the instruction
			auto it = instructions.find(iid);
			// Store the transition
			lts_transitions[current_trans_index].target = tgt;
			lts_transitions[current_trans_index].instruction = it;
			// Store the source state info
			if (first_trans) {
				lts_states[current_state_index].outgoing_begin = current_trans_index;
			}
			current_trans_index++;
		}
		// Store outgoing transitions end for final state
		lts_states[current_state_index].outgoing_end = current_trans_index;
		// for (auto i : PR) {
		// 	for (auto j : i.second) {
		// 		cout << "(" << i.first << ", " << j << ")" << endl;
		// 	}
		// }
		ltsfile.close();

		// For each instruction, create an access reorder relation (PPR, subrelation of PR, consisting of the PR-pairs safe under the weak memory model).
		// Initially, this is empty.
		vector<BiRelation> PPR(instructions.size());

		// Set of (instruction position, thread id) pairs of outgoing transitions of a state
		set<pair<int, int>> out;
		// Compute inter-instruction PR and PRinstr
		for (auto s : lts_states) {
			out.clear();
			// Collect info on outgoing transitions
			for (int i = s.outgoing_begin; i < s.outgoing_end; i++) {
				auto instr = (*(lts_transitions[i].instruction)).second;
				out.insert(pair<int, int>(instr.pos, instr.tid));
			}
			// Check successors
			for (int i = s.outgoing_begin; i < s.outgoing_end; i++) {
				int tgt = lts_transitions[i].target;
				auto instr = (*(lts_transitions[i].instruction)).second;
				int instr_id = (*(lts_transitions[i].instruction)).first;
				int pos = instr.pos;
				int tid = instr.tid;
				for (int j = lts_states[tgt].outgoing_begin; j < lts_states[tgt].outgoing_end; j++) {
					int tgt_instr_id = (*(lts_transitions[j].instruction)).first;
					auto tgt_instr = (*(lts_transitions[j].instruction)).second;
					int tgt_pos = tgt_instr.pos;
					int tgt_tid = tgt_instr.tid;
					if (tgt_tid == tid) {
						if (out.find(pair<int, int>(tgt_pos, tgt_tid)) == out.end()) {
							// PR-relate top elements of instr with bottom elements of tgt_instr
							for (auto ta : instr.top_accs) {
								PR.insert(ta, tgt_instr.bottom_accs);
								PRinstr.insert(instr_id, tgt_instr_id);
							}
						}
					}
				}
			}
		}

		// Compute the RF relation for thread-local variables. This is integrated into DP.
		if (weakmemmodel == ARM) {
			set<int> open;
			set<int> closed;
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
			for (int i = 0; i < accesses.size(); i++) {
				if (PRplus.contains(i) && PRplus.contains(k)) {
					auto ii = PRplus.get(i);
					auto ik = PRplus.get(k);
					for (int j = 0; j < accesses.size(); j++) {
						if (ii->second.find(k) != ii->second.end() && ik->second.find(j) != ik->second.end()) {
							PRplus.insert(i, j);
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
			// Using PRplus, store the presence of unsafe PR-selfloops (i.e., self-loops for read accesses)
			for (int ai = 0; ai < accesses.size(); ai++) {
				if (PRplus.contains(ai)) {
					auto it = PRplus.get(ai);
					if (it->second.find(ai) != it->second.end()) {
						if (accesses.get(ai).type == READ) {
							PRunsafe_selfloops.insert(ai);
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
					reorder(ai, instr.first, instructions, weakmemmodel, PPR, PR, DP, CTRL, false);
					if (PR.contains(ai)) {
						auto it = PR.get(ai);
						for (int bi : it->second) {
							if (instr.second.accesses.find(bi) != instr.second.accesses.end()) {
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
							if (nextset.find(bi) == nextset.end()) {
								nextset.insert(bi);
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
									int result = reorder(ai, instr_id, instructions, weakmemmodel, PPR, PR, DP, CTRL, true);
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

		set<pair<int, int>> pairset;
		// Compute CMP, using the reordering information in the instructions
		for (auto s : lts_states) {
			// Compare outgoing instructions of different threads
			for (int i = s.outgoing_begin; i < s.outgoing_end; i++) {
				auto instr_i = (*(lts_transitions[i].instruction)).second;
				int tid_i = instr_i.tid;
				for (int j = i; j < s.outgoing_end; j++) {
					auto instr_j = (*(lts_transitions[j].instruction)).second;
					int tid_j = instr_j.tid;
					if (tid_i != tid_j) {
						for (int ai : instr_i.accesses) {
							for (int bi : instr_j.accesses) {
								Access& a = accesses.get(ai);
								Access& b = accesses.get(bi);
								if (a.location == b.location && (a.type == WRITE || b.type == WRITE)) {
									if (pairset.find(pair<int, int>(ai, bi)) == pairset.end() && pairset.find(pair<int, int>(bi, ai)) == pairset.end()) {
										pairset.insert(pair<int, int>(ai, bi));
									}
								}
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

		// Sort the CMP vectors stored by the CMP relation by thread ID
		for (int ai = 0; ai < accesses.size(); ai++) {
			sort(CMP.get(ai).begin(), CMP.get(ai).end(), compare_access_indices);
		}

		// Transitively close the PPR relations, via Floyd-Warshall
		for (int w = 0; w < instructions.size(); w++) {
			Instruction& instr = instructions.find(w)->second;
			for (int k = 0; k < instr.accesses.size(); k++) {
				for (int i = 0; i < instr.accesses.size(); i++) {
					if (PPR[w].contains(i) && PPR[w].contains(k)) {
						auto ii = PPR[w].get(i);
						auto ik = PPR[w].get(k);
						for (int j = 0; j < instr.accesses.size(); j++) {
							if (ii->second.find(k) != ii->second.end() && ik->second.find(j) != ik->second.end()) {
								PPR[w].insert(i, j);
							}
						}
					}
				}
			}
		}

		// For each access, create a vector containing int, bool pairs, with the int being IDs of accesses that are PR-reachable,
		// and the bool indicating whether the access can be reached via a PR-unsafe path or not
		vector<vector<pair<int, bool>>> PR_reachable(accesses.size());

		for (int ai = 0; ai < accesses.size(); ai++) {
			Access& a = accesses.get(ai);
			Instruction& instr = instructions.find(a.instruction)->second;
			// Collect the accesses that are PPR reachable inside instr from a
			bool has_PPR_succs = false;
			map<int, set<int>>::iterator PPR_succ_it;
			
			if (PPR[a.instruction].contains(ai)) {
				has_PPR_succs = true;
				PPR_succ_it = PPR[a.instruction].get(ai);
			}
			// Get the PRplus successors of a
			bool has_PRplus_succs = false;
			map<int, set<int>>::iterator PRplus_succ_it;
			
			if (PRplus.contains(ai)) {
				has_PRplus_succs = true;
				PRplus_succ_it = PRplus.get(ai);
			}
			if (has_PRplus_succs) {
				for (auto bi : PRplus_succ_it->second) {
					if (instr.accesses.find(bi) == instr.accesses.end()) {
						PR_reachable[ai].insert(PRsafe_reachable[ai].end(), pair<int, bool>(bi, false));
					}
					else if (has_PPR_succs) {
						if (PPR_succ_it->second.find(bi) != PPR_succ_it->second.end()) {
							PR_reachable[ai].insert(PRsafe_reachable[ai].end(), pair<int, bool>(bi, false));
						}
						else {
							PR_reachable[ai].insert(PRunsafe_reachable[ai].end(), pair<int, bool>(bi, true));
						}
					}
					else {
						PR_reachable[ai].insert(PRunsafe_reachable[ai].end(), pair<int, bool>(bi, true));
					}
				}
				// Sort the final vector based on PR-unsafe reachability (unsafe has priority over safe)
				sort(PR_reachable[ai].begin(), PR_reachable[ai].end(), compare_access_bool_pairs);
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
		vector<bool> mark(accesses.size(), false);
		StaticStack<int> marked_stack(accesses.size());
		StaticStack<StackItem> point_stack(accesses.size());

		StackItem st_tmp;
		bool g;
		bool initial_ai_PR_explored, atomicity_check;
		int unsafe_explored, PR_explored, initial_loc_count;
		set<int> visited_threads;
		set<int> visited_locs;
		set<pair<int, int>> PR_paths_requiring_fences;

		// Count the number of unsafe elements in the program. As soon as everything is marked for fencing, we can stop (early)
		int unsafe_elements_counter;
		for (int i = 0; i < accesses.size(); i++) {
			unsafe_elements_counter += PRunsafe_reachable[i].size();
			// if (RFE.contains(i)) {
			// 	auto it = RFE.get(i);
			// 	unsafe_elements_counter += it->second.size();
			// }
		}

		for (int s = 0; s < accesses.size() && unsafe_elements_counter > 0; s++) {
			if (!accesses.get(s).local) {
				Access& sa = accesses.get(s);
				st_tmp.init(s, 1);
				print(st_tmp.loc_count);
				point_stack.push(st_tmp);
				print(point_stack.peek().loc_count);				
				mark[s] = true;
				marked_stack.push(s);
				initial_ai_PR_explored = false;
				unsafe_explored = 0;
				PR_explored = 0;
				initial_loc_count = 1;
				atomicity_check = false;
				visited_locs.clear();
				visited_locs.insert(sa.location);
				visited_threads.clear();
				visited_threads.insert(sa.tid);
				while (!point_stack.empty() && unsafe_elements_counter > 0) {
					point_stack.print();
					cout << "initial_PR_explored=" << initial_ai_PR_explored << ", unsafe_explored=" << unsafe_explored << ", PR_explored=" << PR_explored << ", initial_loc_count=" << initial_loc_count << endl;
					cout << "visited locations:" << endl;
					for (int i : visited_locs) {
						cout << i << endl;
					}
					cout << "visited threads:" << endl;
					for (int i : visited_threads) {
						cout << i << endl;
					}
					StackItem& v_st = point_stack.peek();
					int w = get_next_edge(v_st, s, initial_loc_count, visited_locs, visited_threads, initial_ai_PR_explored,
								unsafe_explored, PR_explored, atomicity_check, RFE, PRsafe_reachable, PRunsafe_reachable, CMPt, CMP);
					if (w == -1) {
						// Backtrack
						if (v_st.cycle_found) {
							while (marked_stack.peek() != v_st.aid) {
								mark[marked_stack.peek()] = false;
								marked_stack.pop();
							}
							mark[v_st.aid] = false;
							marked_stack.pop();
						}
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
							visited_locs.erase(va.location);
						}
						// If we backtrack over a thread, remove it from the visited threads set
						if (va.tid != sa.tid && va.tid != va_next.tid) {
							visited_threads.erase(va.tid);
						}
						// If we are back at the start, we have not explored a PR-path initially
						if (point_stack.size() == 1) {
							initial_ai_PR_explored = false;
						}
						// Update the number of explored unsafe elements
						if (v_st_next.edge_type == PRUNSAFE || (v_st_next.edge_type == CMPEDGE && RFE.are_related(v_st_next.aid, v_st.aid))) {
							// TODO: improve!
							unsafe_explored--;
						}
						// Update the number of explored PR-paths
						if (v_st_next.edge_type == PRUNSAFE || v_st_next.edge_type == PRSAFE) {
							PR_explored--;
						}
						if (!point_stack.empty()) {
							v_st_next.cycle_found = v_st_next.cycle_found || g;
						}
					}
					else if (w == s) {
						// Process cycle on point stack
						bool mark_next_PR = false;
						for (vector<StackItem>::iterator st = point_stack.begin(); st < point_stack.end(); st++) {
							vector<StackItem>::iterator st_next;
							if (st != point_stack.end()) {
								st_next = st+1;
							}
							else {
								st_next = point_stack.begin();
							}
							if (st->edge_type == PRUNSAFE || (st->edge_type == PRSAFE && mark_next_PR)) {
								PR_paths_requiring_fences.insert(pair<int, int>(st->aid, st_next->aid));
								mark_next_PR = false;
								// Mark the path as safe
								if (st->edge_type == PRUNSAFE) {
									PRsafe_reachable[st->aid].insert(PRsafe_reachable[st->aid].end(), st_next->aid);
									for (int i = 0; i < PRunsafe_reachable[st->aid].size(); i++) {
										if (PRunsafe_reachable[st->aid][i] == st_next->aid) {
											PRunsafe_reachable[st->aid][i] = -1;
											break;
										}
									}
									unsafe_elements_counter--;
								}
							}
							else if (weakmemmodel == ARM && st->edge_type == CMPEDGE) {
								// If we observe an unsafe CMP edge, mark the subsequent PR-path for fencing
								if (RFE.are_related(st->aid, st_next->aid)) {
									mark_next_PR = true;
								}
								// If the previous edge in the cycle is also CMP, check if the access has an unsafe selfloop.
								// If so, that should be marked as well.
								vector<StackItem>::iterator st_prev;
								if (st != point_stack.begin()) {
									st_prev = st-1;
								}
								else {
									st_prev = point_stack.end()-1;
								}
								if (st_prev->edge_type == CMPEDGE && PRunsafe_selfloops.find(st->aid) != PRunsafe_selfloops.end()) {
									PR_paths_requiring_fences.insert(pair<int, int>(st->aid, st->aid));
								}
							}
						}
						// Continue looking for a PR-path to fence, if needed
						if (mark_next_PR) {
							for (vector<StackItem>::iterator st = point_stack.begin(); st < point_stack.end(); st++) {
								vector<StackItem>::iterator st_next;
								if (st != point_stack.begin()) {
									st_next = st+1;
								}
								else {
									st_next = point_stack.begin();
								}
								if (st->edge_type == PRUNSAFE || st->edge_type == PRSAFE) {
									PR_paths_requiring_fences.insert(pair<int, int>(st->aid, st_next->aid));
									break;
								}
							}
						}
						cout << "Cycle!" << endl;
						v_st.cycle_found = true;
						unsafe_explored = 0;
					}
					else if (!mark[w]) {
						int loc_count = 1;
						Access& va = accesses.get(v_st.aid);
						Access& wa = accesses.get(w);
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
						mark[w] = true;
						marked_stack.push(w);
						continue;
					}
				}
				while (!marked_stack.empty()) {
					mark[marked_stack.peek()] = false;
					marked_stack.pop();
				}
			}
		}
		cout << "Marked for fencing:" << endl;
		for (auto p : PR_paths_requiring_fences) {
			cout << p.first << " -PR-> " << p.second << endl;
		}


		// TODO in cycle postprocessing:
		// - under ARM, check whether cycle can be extended with unsafe selfloops (from a read to itself)
		// - mark unsafe PR-paths in cycle as safe (after counting the involved accesses)
		// DURING cycle detection: check whether there are still unsafe PR-paths and/or unsafe CMP-edges left
		// If no, cycle detection can stop

		// cout << "PR:" << endl;
		// for (auto i : PR) {
		// 	for (auto j : i.second) {
		// 		cout << "(" << i.first << ", " << j << ")" << endl;
		// 	}
		// }
		// cout << "The PPRs: " << endl;
		// for (int i = 0; i < instructions.size(); i++) {
		// 	for (auto i : PPR[i]) {
		// 		for (auto j : i.second) {
		// 			cout << "(" << i.first << ", " << j << ")" << endl;
		// 		}
		// 	}
		// }
		cout << "PRplus:" << endl;
		for (auto i : PRplus) {
			for (auto j : i.second) {
				cout << "(" << i.first << ", " << j << ")" << endl;
			}
		}
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
		cout << "CMP:" << endl;
		for (int i = 0; i < CMP.size(); i++) {
			vector<int>& S = CMP.get(i);
			for (auto j : S) {
				cout << "(" << i << ", " << j << ")" << endl;
			}
		}
		// cout << "CMPt:" << endl;
		// for (int i = 0; i < CMPt.size(); i++) {
		// 	vector<ThreadAccessRange>& S = CMPt.get(i);
		// 	for (ThreadAccessRange j : S) {
		// 		cout << "(" << i << ", " << j.tid << ", " << j.accesses_begin << ", " << j.accesses_end << ")" << endl;
		// 	}
		// }	
	}
	else {
		cout << "LTS file does not exist!" << endl;
		exit(-1);
	}
}
