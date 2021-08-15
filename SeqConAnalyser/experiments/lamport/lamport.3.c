int x;
int y;
int b1, b2, b3, b4; // N boolean flags

void* thr1(void * arg) {
	while(1) {
		// SL0
		b1 = 1;
		// SimpleState0
		x = 1;
		// S0
		if (y != 0) {
			// SimpleState1
			b1 = 0;
			continue;
			// -> SL0
		}
		else {
			y = 1;
			// S1
			if (x == 1) {
				// S3
				y = 0;
				// SimpleState2
			SimpleState2:
				b1 = 0;
				break;
				// -> STOP
			}
			else {
				b1 = 0;
				// S2
				if (y != 1) {
					continue;
					// -> SL0
				}
				else {
					y = 0;
					// SimpleState2
					goto SimpleState2;
				}
			}
		}
  }
}

void* thr2(void * arg) {
	while(1) {
		// SL0
		b2 = 1;
		// SimpleState0
		x = 2;
		// S0
		if (y != 0) {
			// SimpleState1
			b2 = 0;
			continue;
			// -> SL0
		}
		else {
			y = 2;
			// S1
			if (x == 2) {
				// S3
				y = 0;
				// SimpleState2
			SimpleState2:
				b2 = 0;
				break;
				// -> STOP
			}
			else {
				b2 = 0;
				// S2
				if (y != 2) {
					continue;
					// -> SL0
				}
				else {
					y = 0;
					// SimpleState2
					goto SimpleState2;
				}
			}
		}
  }
}

void* thr3(void * arg) {
	while(1) {
		// SL0
		b3 = 1;
		// SimpleState0
		x = 3;
		// S0
		if (y != 0) {
			// SimpleState1
			b3 = 0;
			continue;
			// -> SL0
		}
		else {
			y = 3;
			// S1
			if (x == 3) {
				// S3
				y = 0;
				// SimpleState2
			SimpleState2:
				b3 = 0;
				break;
				// -> STOP
			}
			else {
				b3 = 0;
				// S2
				if (y != 3) {
					continue;
					// -> SL0
				}
				else {
					y = 0;
					// SimpleState2
					goto SimpleState2;
				}
			}
		}
  }
}

void* thr4(void * arg) {
	while(1) {
		// SL0
		b4 = 1;
		// SimpleState0
		x = 4;
		// S0
		if (y != 0) {
			// SimpleState1
			b4 = 0;
			continue;
			// -> SL0
		}
		else {
			y = 4;
			// S1
			if (x == 4) {
				// S3
				y = 0;
				// SimpleState2
			SimpleState2:
				b4 = 0;
				break;
				// -> STOP
			}
			else {
				b4 = 0;
				// S2
				if (y != 4) {
					continue;
					// -> SL0
				}
				else {
					y = 0;
					// SimpleState2
					goto SimpleState2;
				}
			}
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
