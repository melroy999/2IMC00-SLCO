// #include <assert.h>

int b[3] = {0,0,2};
int readers;
int sem = 1;
int changeto;

void* slave1(void * arg) {
	// idle
	while (1) {
		__CPROVER_assume (sem == 1);
		sem = 0;
		// q0
		if (b[0] == 1) {
			b[0] = 0;
		}
		else if (b[1] == 1) {
			b[1] = 0;
		}
		else if (b[2] == 1) {
			b[2] = 0;
		}
		// q1
		if (b[0] == 2) {
			b[0] = 1;
		}
		else if (b[1] == 2) {
			b[1] = 1;
		}
		else if (b[2] == 2) {
			b[2] = 1;
		}
		else {
			break;
			// -> errorstate
		}
		// q2
		if (b[0] == 0) {
			b[0] = 2;
		}
		else if (b[1] == 0) {
			b[1] = 2;
		}
		else if (b[2] == 0) {
			b[2] = 2;
		}
		else {
			break;
			// -> errorstate
		}
		// q3
		sem = 1;
	}
}

void* slave2(void * arg) {
	// idle
	while (1) {
		__CPROVER_assume (sem == 1);
		sem = 0;
		// q0
		if (b[0] == 1) {
			b[0] = 0;
		}
		else if (b[1] == 1) {
			b[1] = 0;
		}
		else if (b[2] == 1) {
			b[2] = 0;
		}
		// q1
		if (b[0] == 2) {
			b[0] = 1;
		}
		else if (b[1] == 2) {
			b[1] = 1;
		}
		else if (b[2] == 2) {
			b[2] = 1;
		}
		else {
			break;
			// -> errorstate
		}
		// q2
		if (b[0] == 0) {
			b[0] = 2;
		}
		else if (b[1] == 0) {
			b[1] = 2;
		}
		else if (b[2] == 0) {
			b[2] = 2;
		}
		else {
			break;
			// -> errorstate
		}
		// q3
		sem = 1;
	}
}

void* master1(void * arg) {
	int tmp;
	int tmp1;
	
	// idle
	while (1) {
		__CPROVER_assume (sem == 1);
		sem = 0;
		// q0
		tmp = b[0];
		tmp1 = b[1];
		if (tmp == 3 || tmp1 == 3 || b[2] == 3) {
			// -> master
		}
		else {
			// nomaster
			if (b[0] == 1) {
				b[0] = 3;
				// -> master
			}
			else if (b[1] == 1) {
				b[1] = 3;
				// -> master
			}
			else if (b[2] == 1) {
				b[2] = 3;
				// -> master
			}
			else {
				sem = 1;
				continue;
				// -> idle
			}
		}
		// master
		readers = readers+1;
		sem = 1;
		// reading
		__CPROVER_assume (sem == 1);
		sem = 0;
		readers = readers-1;
		// r0
		if (readers == 0) {
			// noreaders
			if (b[0] == 1) {
				changeto = 0;
				// -> change
			}
			else {
				changeto = 1;
				// -> change
			}
			// change
			__CPROVER_assume (b[0] == 3);
			b[0] = changeto;
			// -> r1
		}
		// r1
		sem = 1;
	}
}

void* master2(void * arg) {
	int tmp;
	int tmp1;
	
	// idle
	while (1) {
		__CPROVER_assume (sem == 1);
		sem = 0;
		// q0
		tmp = b[0];
		tmp1 = b[1];
		if (tmp == 3 || tmp1 == 3 || b[2] == 3) {
			// -> master
		}
		else {
			// nomaster
			if (b[0] == 1) {
				b[0] = 3;
				// -> master
			}
			else if (b[1] == 1) {
				b[1] = 3;
				// -> master
			}
			else if (b[2] == 1) {
				b[2] = 3;
				// -> master
			}
			else {
				sem = 1;
				continue;
				// -> idle
			}
		}
		// master
		readers = readers+1;
		sem = 1;
		// reading
		__CPROVER_assume (sem == 1);
		sem = 0;
		readers = readers-1;
		// r0
		if (readers == 0) {
			// noreaders
			if (b[0] == 1) {
				changeto = 0;
				// -> change
			}
			else {
				changeto = 1;
				// -> change
			}
			// change
			__CPROVER_assume (b[0] == 3);
			b[0] = changeto;
			// -> r1
		}
		// r1
		sem = 1;
	}
}

void* master3(void * arg) {
	int tmp;
	int tmp1;
	
	// idle
	while (1) {
		__CPROVER_assume (sem == 1);
		sem = 0;
		// q0
		tmp = b[0];
		tmp1 = b[1];
		if (tmp == 3 || tmp1 == 3 || b[2] == 3) {
			// -> master
		}
		else {
			// nomaster
			if (b[0] == 1) {
				b[0] = 3;
				// -> master
			}
			else if (b[1] == 1) {
				b[1] = 3;
				// -> master
			}
			else if (b[2] == 1) {
				b[2] = 3;
				// -> master
			}
			else {
				sem = 1;
				continue;
				// -> idle
			}
		}
		// master
		readers = readers+1;
		sem = 1;
		// reading
		__CPROVER_assume (sem == 1);
		sem = 0;
		readers = readers-1;
		// r0
		if (readers == 0) {
			// noreaders
			if (b[0] == 1) {
				changeto = 0;
				// -> change
			}
			else {
				changeto = 1;
				// -> change
			}
			// change
			__CPROVER_assume (b[0] == 3);
			b[0] = changeto;
			// -> r1
		}
		// r1
		sem = 1;
	}
}

int main()
{
  __CPROVER_ASYNC_1: slave1(0);
  __CPROVER_ASYNC_2: slave2(0);
  __CPROVER_ASYNC_3: master1(0);
  __CPROVER_ASYNC_4: master2(0);
	master3(0);
}
