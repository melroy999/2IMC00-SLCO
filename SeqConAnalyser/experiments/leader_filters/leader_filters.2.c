// #include <assert.h>

int turn[5];
int b[5];
int c[5];

void* P0(void * arg) {
	int curr = 0;
	
	// p1
	while (1) {
		turn[curr] = 0;
		// p2
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

int main()
{
  __CPROVER_ASYNC_1: P0(0);
  __CPROVER_ASYNC_2: P1(0);
	P2(0);
}
