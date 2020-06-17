int x;
int y;
int b1, b2, b3; // N boolean flags

void* thr1(void * arg) {
  L0:
    b1 = 1;
    x = 1;
    if (y != 0) {
      b1 = 0;
      __CPROVER_assume(y == 0);
      goto L0;
    }
    y = 1;
    if (x != 1) {
      b1 = 0;
      __CPROVER_assume (b2 == 0 && b3 == 0);
      if (y != 1) {
	__CPROVER_assume (y == 0);
	goto L0;
      }
    }
  // begin: critical section
  // end: critical section
  y = 0;
  b1 = 0;
}

void* thr2(void * arg) {
  L1:
    b2 = 1;
    x = 2;
    if (y != 0) {
      b2 = 0;
      __CPROVER_assume (y == 0);
      goto L1;
    }
    y = 2;
    if (x != 2) {
      b2 = 0;
      __CPROVER_assume (b1 == 0 && b3 == 0);
      if (y != 2) {
	__CPROVER_assume (y == 0);
	goto L1;
      }
    }
  // begin: critical section
  // end: critical section
  y = 0;
  b2 = 0;
}

void* thr3(void * arg) {
  L2:
    b3 = 1;
    x = 3;
    if (y != 0) {
      b3 = 0;
      __CPROVER_assume (y == 0);
      goto L2;
    }
    y = 3;
    if (x != 3) {
      b3 = 0;
      __CPROVER_assume (b1 == 0 && b2 == 0);
      if (y != 3) {
    __CPROVER_assume (y == 0);
    goto L2;
      }
    }
  // begin: critical section
  // end: critical section
  y = 0;
  b3 = 0;
}

int main()
{
  __CPROVER_ASYNC_1: thr1(0);
  thr2(0);
  thr3(0);
}
