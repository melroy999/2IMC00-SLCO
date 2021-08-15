// #include <assert.h>

int turn[6];
int b[6];
int c[6];

void* P0(void * arg) {
	int curr = 0;
	
	// p1
	while (1) {
		turn[curr] = 0;
		// p2
		__CPROVER_assume (b[curr] == 0);
		// p3
		b[curr] = 1;
		// p4
		if (turn[curr] == 0) {
			// p8
			if (curr == 0) {
				curr = curr+1;
				continue;
				// -> p1
			}
			else if (c[curr-1] == 0) {
				break;
				// -> elected
			}
			else {
				curr = curr+1;
				continue;
				// -> p1
			}
		}
		else {
			c[curr] = 1;
			// p6
			b[curr] = 0;
			break;
			// -> p7
		}
	}
}

void* P1(void * arg) {
	int curr = 0;
	
	// p1
	while (1) {
		turn[curr] = 1;
		// p2
		__CPROVER_assume (b[curr] == 0);
		// p3
		b[curr] = 1;
		// p4
		if (turn[curr] == 1) {
			// p8
			if (curr == 0) {
				curr = curr+1;
				continue;
				// -> p1
			}
			else if (c[curr-1] == 0) {
				break;
				// -> elected
			}
			else {
				curr = curr+1;
				continue;
				// -> p1
			}
		}
		else {
			c[curr] = 1;
			// p6
			b[curr] = 0;
			break;
			// -> p7
		}
	}
}

void* P2(void * arg) {
	int curr = 0;
	
	// p1
	while (1) {
		turn[curr] = 2;
		// p2
		__CPROVER_assume (b[curr] == 0);
		// p3
		b[curr] = 1;
		// p4
		if (turn[curr] == 2) {
			// p8
			if (curr == 0) {
				curr = curr+1;
				continue;
				// -> p1
			}
			else if (c[curr-1] == 0) {
				break;
				// -> elected
			}
			else {
				curr = curr+1;
				continue;
				// -> p1
			}
		}
		else {
			c[curr] = 1;
			// p6
			b[curr] = 0;
			break;
			// -> p7
		}
	}
}

void* P3(void * arg) {
	int curr = 0;
	
	// p1
	while (1) {
		turn[curr] = 3;
		// p2
		__CPROVER_assume (b[curr] == 0);
		// p3
		b[curr] = 1;
		// p4
		if (turn[curr] == 3) {
			// p8
			if (curr == 0) {
				curr = curr+1;
				continue;
				// -> p1
			}
			else if (c[curr-1] == 0) {
				break;
				// -> elected
			}
			else {
				curr = curr+1;
				continue;
				// -> p1
			}
		}
		else {
			c[curr] = 1;
			// p6
			b[curr] = 0;
			break;
			// -> p7
		}
	}
}

void* P4(void * arg) {
	int curr = 0;
	
	// p1
	while (1) {
		turn[curr] = 4;
		// p2
		__CPROVER_assume (b[curr] == 0);
		// p3
		b[curr] = 1;
		// p4
		if (turn[curr] == 4) {
			// p8
			if (curr == 0) {
				curr = curr+1;
				continue;
				// -> p1
			}
			else if (c[curr-1] == 0) {
				break;
				// -> elected
			}
			else {
				curr = curr+1;
				continue;
				// -> p1
			}
		}
		else {
			c[curr] = 1;
			// p6
			b[curr] = 0;
			break;
			// -> p7
		}
	}
}

void* P5(void * arg) {
	int curr = 0;
	
	// p1
	while (1) {
		turn[curr] = 5;
		// p2
		__CPROVER_assume (b[curr] == 0);
		// p3
		b[curr] = 1;
		// p4
		if (turn[curr] == 5) {
			// p8
			if (curr == 0) {
				curr = curr+1;
				continue;
				// -> p1
			}
			else if (c[curr-1] == 0) {
				break;
				// -> elected
			}
			else {
				curr = curr+1;
				continue;
				// -> p1
			}
		}
		else {
			c[curr] = 1;
			// p6
			b[curr] = 0;
			break;
			// -> p7
		}
	}
}

int main()
{
  __CPROVER_ASYNC_1: P0(0);
  __CPROVER_ASYNC_2: P1(0);
	__CPROVER_ASYNC_3: P2(0);
	__CPROVER_ASYNC_4: P3(0);
	__CPROVER_ASYNC_5: P4(0);
	P5(0);
}
