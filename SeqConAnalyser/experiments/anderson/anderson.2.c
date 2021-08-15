// #include <assert.h>

int Slot[3] = {1,0,0};
int next = 0;

void* P0(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 3-1) {
				next = next-3;
			}
			else {
				myplace = myplace % 3;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%3] = 1;
	 }
}

void* P1(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 3-1) {
				next = next-3;
			}
			else {
				myplace = myplace % 3;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%3] = 1;
	 }
}

void* P2(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 3-1) {
				next = next-3;
			}
			else {
				myplace = myplace % 3;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%3] = 1;
	 }
}

int main()
{
  __CPROVER_ASYNC_1: P0(0);
  __CPROVER_ASYNC_2: P1(0);
  P2(0);
}
