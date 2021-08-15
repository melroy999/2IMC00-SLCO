// #include <assert.h>

int flag1 = 0, flag2 = 0, flag3 = 0, flag4 = 0; // N boolean flags
int turn = 0; // integer variable to hold the ID of the thread whose turn is it
//int x; // variable to test mutual exclusion

void* thr1(void * arg) {
	int ftmp, ftmp2;
	// S0
  flag1 = 1;
  // S1
  while(1) {
		ftmp = flag3;
		ftmp2 = flag4;
		if (flag2 == 1 || ftmp == 1 || ftmp2 == 1) {
			// S2
			if (turn == 0) {
				continue;
				// -> S1
			}
			else {
				flag1 = 0;
				// S3
				__CPROVER_assume (turn == 0);
				flag1 = 1;
				continue;
				// -> S1
			}
		}
		else {
			turn = 1;
			// SimpleState
			flag1 = 0;
			break;
			// -> STOP
		}
	}
}

void* thr2(void * arg) {
	int ftmp, ftmp2;
	// S0
  flag2 = 1;
  // S1
  while(1) {
		ftmp = flag3;
		ftmp2 = flag4;
		if (flag1 == 1 || ftmp == 1 || ftmp2 == 1) {
			// S2
			if (turn == 1) {
				continue;
				// -> S1
			}
			else {
				flag2 = 0;
				// S3
				__CPROVER_assume (turn == 1);
				flag2 = 1;
				continue;
				// -> S1
			}
		}
		else {
			turn = 2;
			// SimpleState
			flag2 = 0;
			break;
			// -> STOP
		}
	}
}

void* thr3(void * arg) {
	int ftmp, ftmp2;
	// S0
  flag3 = 1;
  // S1
  while(1) {
		ftmp = flag2;
		ftmp2 = flag4;
		if (flag1 == 1 || ftmp == 1 || ftmp2 == 1) {
			// S2
			if (turn == 2) {
				continue;
				// -> S1
			}
			else {
				flag3 = 0;
				// S3
				__CPROVER_assume (turn == 2);
				flag3 = 1;
				continue;
				// -> S1
			}
		}
		else {
			turn = 3;
			// SimpleState
			flag3 = 0;
			break;
			// -> STOP
		}
	}
}

void* thr4(void * arg) {
	int ftmp, ftmp2;
	// S0
  flag4 = 1;
  // S1
  while(1) {
		ftmp = flag2;
		ftmp2 = flag3;
		if (flag1 == 1 || ftmp == 1 || ftmp2 == 1) {
			// S2
			if (turn == 3) {
				continue;
				// -> S1
			}
			else {
				flag4 = 0;
				// S3
				__CPROVER_assume (turn == 3);
				flag4 = 1;
				continue;
				// -> S1
			}
		}
		else {
			turn = 0;
			// SimpleState
			flag4 = 0;
			break;
			// -> STOP
		}
	}
}

int main()
{
  __CPROVER_ASYNC_1: thr1(0);
  __CPROVER_ASYNC_2: thr2(0);
  __CPROVER_ASYNC_3: thr3(0);
  thr4(0);
}
