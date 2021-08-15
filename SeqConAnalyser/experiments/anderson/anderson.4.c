// #include <assert.h>

int Slot[4] = {1,0,0,0};
int next = 0;

void* P0(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 4-1) {
				next = next-4;
			}
			else {
				myplace = myplace % 4;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%4] = 1;
	 }
}

void* P1(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 4-1) {
				next = next-4;
			}
			else {
				myplace = myplace % 4;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%4] = 1;
	 }
}

void* P2(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 4-1) {
				next = next-4;
			}
			else {
				myplace = myplace % 4;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%4] = 1;
	 }
}

void* P3(void * arg) {
    int myplace;

    // NCS
    while (1) {
			myplace = next;
			next = next+1;
			// p1
			if (myplace == 4-1) {
				next = next-4;
			}
			else {
				myplace = myplace % 4;
			}
			// p2
			__CPROVER_assume (Slot[myplace] == 1);
			// p3
			Slot[myplace] = 0;
			// CS
			Slot[(myplace+1)%4] = 1;
	 }
}

int main()
{
  __CPROVER_ASYNC_1: P0(0);
  __CPROVER_ASYNC_2: P1(0);
  __CPROVER_ASYNC_3: P2(0);
  P3(0);
}
