// #include <assert.h>

int req[10];
int t;
int p;
int v;

void* cabin(void * arg) {
	int tmp;

	// idle
	while (1) {
		__CPROVER_assume (v>0);
		// mov
		while (1) {
			tmp = t;
			if (tmp < p) {
				p = p-1;
			}
			else if (tmp > p) {
				p = p+1;
			}
			else {
				req[p] = 0;
				v = 0;
				break;
			}
		}
	}
}

void* environment(void * arg) {
	// read
	while (1) {
		if (req[0] == 0) {
			req[0] = 1;
		}
		if (req[1] == 0) {
			req[1] = 1;
		}
		if (req[2] == 0) {
			req[2] = 1;
		}
		if (req[3] == 0) {
			req[3] = 1;
		}
		if (req[4] == 0) {
			req[4] = 1;
		}
		if (req[5] == 0) {
			req[5] = 1;
		}
		if (req[6] == 0) {
			req[6] = 1;
		}
		if (req[7] == 0) {
			req[7] = 1;
		}
		if (req[8] == 0) {
			req[8] = 1;
		}
		if (req[9] == 0) {
			req[9] = 1;
		}
	}
}

void* controller(void * arg) {
	int ldir = 0;

	// wait
	while (1) {
		__CPROVER_assume (v == 0);
		// work
		if (req[0] == 1) {
			t = 0;
		}
		if (req[1] == 1) {
			t = 1;
		}
		if (req[2] == 1) {
			t = 2;
		}
		if (req[3] == 1) {
			t = 3;
		}
		if (req[4] == 1) {
			t = 4;
		}
		if (req[5] == 1) {
			t = 5;
		}
		if (req[6] == 1) {
			t = 6;
		}
		if (req[7] == 1) {
			t = 7;
		}
		if (req[8] == 1) {
			t = 8;
		}
		if (req[9] == 1) {
			t = 9;
		}
		v = 1;
	}
}

int main()
{
  __CPROVER_ASYNC_1: cabin(0);
  __CPROVER_ASYNC_2: environment(0);
	controller(0);
}
