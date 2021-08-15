// #include <assert.h>

int Slot[6] = {1,0,0,0,0,0};
int next = 0;

void* P0(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 6-1) {
				next = next-6;
			}
			else {
				myplace = myplace % 6;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%6] = 1;
	 }
}

void* P1(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 6-1) {
				next = next-6;
			}
			else {
				myplace = myplace % 6;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%6] = 1;
	 }
}

void* P2(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 6-1) {
				next = next-6;
			}
			else {
				myplace = myplace % 6;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%6] = 1;
	 }
}

void* P3(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 6-1) {
				next = next-6;
			}
			else {
				myplace = myplace % 6;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%6] = 1;
	 }
}

void* P4(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 6-1) {
				next = next-6;
			}
			else {
				myplace = myplace % 6;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%6] = 1;
	 }
}

void* P5(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 6-1) {
				next = next-6;
			}
			else {
				myplace = myplace % 6;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%6] = 1;
	 }
}

int main()
{
  __CPROVER_ASYNC_1: P0(0);
  __CPROVER_ASYNC_2: P1(0);
  __CPROVER_ASYNC_3: P2(0);
  __CPROVER_ASYNC_3: P3(0);
  __CPROVER_ASYNC_4: P4(0);
  P5(0);
}
