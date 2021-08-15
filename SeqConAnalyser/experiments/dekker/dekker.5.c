// #include <assert.h>

int flag1 = 0, flag2 = 0, flag3 = 0, flag4 = 0, flag5 = 0, flag6 = 0; // N boolean flags
int turn = 0; // integer variable to hold the ID of the thread whose turn is it
//int x; // variable to test mutual exclusion

void* thr1(void * arg) {
	int ftmp, ftmp2, ftmp3, ftmp4;
	// S0
  flag1 = 1;
  // S1
  while(1) {
		ftmp = flag3;
		ftmp2 = flag4;
		ftmp3 = flag5;
		ftmp4 = flag6;
		if (flag2 == 1 || ftmp == 1 || ftmp2 == 1 || ftmp3 == 1 || ftmp4 == 1) {
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
	int ftmp, ftmp2, ftmp3, ftmp4;
	// S0
  flag2 = 1;
  // S1
  while(1) {
		ftmp = flag3;
		ftmp2 = flag4;
		ftmp3 = flag5;
		ftmp4 = flag6;
		if (flag1 == 1 || ftmp == 1 || ftmp2 == 1 || ftmp3 == 1 || ftmp4 == 1) {
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
	int ftmp, ftmp2, ftmp3, ftmp4;
	// S0
  flag3 = 1;
  // S1
  while(1) {
		ftmp = flag2;
		ftmp2 = flag4;
		ftmp3 = flag5;
		ftmp4 = flag6;
		if (flag1 == 1 || ftmp == 1 || ftmp2 == 1 || ftmp3 == 1 || ftmp4 == 1) {
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
	int ftmp, ftmp2, ftmp3, ftmp4;
	// S0
  flag4 = 1;
  // S1
  while(1) {
		ftmp = flag2;
		ftmp2 = flag3;
		ftmp3 = flag5;
		ftmp4 = flag6;
		if (flag1 == 1 || ftmp == 1 || ftmp2 == 1 || ftmp3 == 1 || ftmp4 == 1) {
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
			turn = 4;
			// SimpleState
			flag4 = 0;
			break;
			// -> STOP
		}
	}
}

void* thr5(void * arg) {
	int ftmp, ftmp2, ftmp3, ftmp4;
	// S0
  flag5 = 1;
  // S1
  while(1) {
		ftmp = flag2;
		ftmp2 = flag3;
		ftmp3 = flag4;
		ftmp4 = flag6;
		if (flag1 == 1 || ftmp == 1 || ftmp2 == 1 || ftmp3 == 1 || ftmp4 == 1) {
			// S2
			if (turn == 4) {
				continue;
				// -> S1
			}
			else {
				flag5 = 0;
				// S3
				__CPROVER_assume (turn == 4);
				flag5 = 1;
				continue;
				// -> S1
			}
		}
		else {
			turn = 5;
			// SimpleState
			flag5 = 0;
			break;
			// -> STOP
		}
	}
}

void* thr6(void * arg) {
	int ftmp, ftmp2, ftmp3, ftmp4;
	// S0
  flag6 = 1;
  // S1
  while(1) {
		ftmp = flag2;
		ftmp2 = flag3;
		ftmp3 = flag4;
		ftmp4 = flag5;
		if (flag1 == 1 || ftmp == 1 || ftmp2 == 1 || ftmp3 == 1 || ftmp4 == 1) {
			// S2
			if (turn == 5) {
				continue;
				// -> S1
			}
			else {
				flag6 = 0;
				// S3
				__CPROVER_assume (turn == 5);
				flag6 = 1;
				continue;
				// -> S1
			}
		}
		else {
			turn = 0;
			// SimpleState
			flag6 = 0;
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
  __CPROVER_ASYNC_4: thr4(0);
  __CPROVER_ASYNC_5: thr5(0);
  thr6(0);
}
