// #include <assert.h>

int next[5] = {255,255,255,255,255};
int locked[5];
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
			__CPROVER_assume (locked[0] == 0);
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
				__CPROVER_assume (next[0] != 255);
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
			__CPROVER_assume (locked[1] == 0);
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
				__CPROVER_assume (next[1] != 255);
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
			__CPROVER_assume (locked[2] == 0);
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
				__CPROVER_assume (next[2] != 255);
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

void* P3(void * arg) {
	int pred;
	int tmp;
	
	// NCS
	while (1) {
		next[3] = 255;
		// p2
		pred = tail;
		tail = 3;
		// p3
		if (pred == 255) {
			// -> CS
		}
		else {
			locked[3] = 1;
			// p5
			next[pred] = 3;
			// p6
			__CPROVER_assume (locked[3] == 0);
			// -> CS
		}
		// CS
		if (next[3] == 255) {
			// p9
			if (tail == 3) {
				tail = 255;
				// -> NCS
			}
			else {
				__CPROVER_assume (next[3] != 255);
				// p13
				tmp = next[3];
				locked[tmp] = 0;
				// -> NCS
			}
		}
		else {
			tmp = next[3];
			locked[tmp] = 0;
			// -> NCS
		}
	}
}

void* P4(void * arg) {
	int pred;
	int tmp;
	
	// NCS
	while (1) {
		next[4] = 255;
		// p2
		pred = tail;
		tail = 4;
		// p3
		if (pred == 255) {
			// -> CS
		}
		else {
			locked[4] = 1;
			// p5
			next[pred] = 4;
			// p6
			__CPROVER_assume (locked[4] == 0);
			// -> CS
		}
		// CS
		if (next[4] == 255) {
			// p9
			if (tail == 4) {
				tail = 255;
				// -> NCS
			}
			else {
				__CPROVER_assume (next[4] != 255);
				// p13
				tmp = next[4];
				locked[tmp] = 0;
				// -> NCS
			}
		}
		else {
			tmp = next[4];
			locked[tmp] = 0;
			// -> NCS
		}
	}
}

int main()
{
  __CPROVER_ASYNC_1: P0(0);
  __CPROVER_ASYNC_2: P1(0);
  __CPROVER_ASYNC_3: P2(0);
  __CPROVER_ASYNC_4: P3(0);
  P4(0);
}
