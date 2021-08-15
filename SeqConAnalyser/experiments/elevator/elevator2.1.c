// #include <assert.h>

int req[4];
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
	}
}

void* controller(void * arg) {
	int ldir = 0;
	int tmp;

	// wait
	while (1) {
		__CPROVER_assume (v == 0);
		t = t+(2*ldir)-1;
		// work
		while (1) {
			tmp = t;
			if (tmp < 0 || tmp == 4) {
				ldir = 1-ldir;
				break;
				// -> wait
			}
			else if (req[tmp] == 1) {
				// done
				v = 1;
				break;
				// -> wait
			}
			else {
				t = t+(2*ldir)-1;
			}
		}
		break;
	}
}

int main()
{
  __CPROVER_ASYNC_1: cabin(0);
  __CPROVER_ASYNC_2: environment(0);
	controller(0);
}
