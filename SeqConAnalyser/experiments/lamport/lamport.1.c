int x;
int y;
int b1, b2; // N boolean flags

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

int main()
{
  __CPROVER_ASYNC_1: thr1(0);
  thr2(0);
}
