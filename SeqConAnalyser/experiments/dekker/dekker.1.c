// #include <assert.h>

int flag1 = 0, flag2 = 0; // N boolean flags 
int turn = 0; // integer variable to hold the ID of the thread whose turn is it
//int x; // variable to test mutual exclusion

void* thr1(void * arg) {
	// S0
  flag1 = 1;
  // S1
  while(1) {
		if (flag2 == 1) {
			// S2
			if (turn == 0) {
				continue;
				// -> S1
			}
			else {
				flag1 = 0;
				// S3
				__CPROVER_assume (turn != 1);
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
	// S0
  flag2 = 1;
  // S1
  while(1) {
		if (flag1 == 1) {
			// S2
			if (turn == 1) {
				continue;
				// -> S1
			}
			else {
				flag2 = 0;
				// S3
				__CPROVER_assume (turn != 0);
				flag2 = 1;
				continue;
				// -> S1
			}
		}
		else {
			turn = 0;
			// SimpleState
			flag2 = 0;
			break;
			// -> STOP
		}
	}
}

int main()
{
  __CPROVER_ASYNC_1: thr1(0);
  thr2(0);
}
