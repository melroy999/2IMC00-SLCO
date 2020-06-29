// #include <assert.h>

int next[3] = {255,255,255};
int locked[3];
int tail = 255;

void* P0(void * arg) {
	int pred;
	int tmp;
	
	// NCS
	while (1) {
		next[0] = 255;
		// p2
		pred = tail;
		tail = 0;
		// p3
		if (pred == 255) {
			// -> CS
		}
		else {
			locked[0] = 1;
			// p5
			next[pred] = 0;
			// p6
			while (locked[0] != 0) {};
			// -> CS
		}
		// CS
		if (next[0] == 255) {
			// p9
			if (tail == 0) {
				tail = 255;
				// -> NCS
			}
			else {
				while (next[0] == 255) {};
				// p13
				tmp = next[0];
				locked[tmp] = 0;
				// -> NCS
			}
		}
		else {
			tmp = next[0];
			locked[tmp] = 0;
			// -> NCS
		}
	}
}

void* P1(void * arg) {
	int pred;
	int tmp;
	
	// NCS
	while (1) {
		next[1] = 255;
		// p2
		pred = tail;
		tail = 1;
		// p3
		if (pred == 255) {
			// -> CS
		}
		else {
			locked[1] = 1;
			// p5
			next[pred] = 1;
			// p6
			while (locked[1] != 0) {};
			// -> CS
		}
		// CS
		if (next[1] == 255) {
			// p9
			if (tail == 1) {
				tail = 255;
				// -> NCS
			}
			else {
				while (next[1] == 255) {};
				// p13
				tmp = next[1];
				locked[tmp] = 0;
				// -> NCS
			}
		}
		else {
			tmp = next[1];
			locked[tmp] = 0;
			// -> NCS
		}
	}
}

void* P2(void * arg) {
	int pred;
	int tmp;
	
	// NCS
	while (1) {
		next[2] = 255;
		// p2
		pred = tail;
		tail = 2;
		// p3
		if (pred == 255) {
			// -> CS
		}
		else {
			locked[2] = 1;
			// p5
			next[pred] = 2;
			// p6
			while (locked[2] != 0) {};
			// -> CS
		}
		// CS
		if (next[2] == 255) {
			// p9
			if (tail == 2) {
				tail = 255;
				// -> NCS
			}
			else {
				while (next[2] == 255) {};
				// p13
				tmp = next[2];
				locked[tmp] = 0;
				// -> NCS
			}
		}
		else {
			tmp = next[2];
			locked[tmp] = 0;
			// -> NCS
		}
	}
}

int main()
{
  __CPROVER_ASYNC_1: P0(0);
  __CPROVER_ASYNC_2: P1(0);
	P2(0);
}
