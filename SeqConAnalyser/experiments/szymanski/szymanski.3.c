
int flag1 = 0, flag2 = 0, flag3 = 0, flag4 = 0; // N integer flags

void* thr1(void * arg) {
  while (1) {
    flag1 = 1;
    while (!(flag2 < 3 && flag3 < 3 && flag4 < 3)) {};
    flag1 = 3;
    if (flag2 == 1 || flag3 == 1 || flag4 == 1) {
      flag1 = 2;
      while (!(flag2 == 4 || flag3 == 4 || flag4 == 4)) {};
    }
    flag1 = 4;
    while (!(flag2 < 2 && flag3 < 2 && flag4 < 2)) {};
    // begin critical section
    // end critical section
    while (!((2 > flag2 || flag2 > 3) && (2 > flag3 || flag3 > 3) && (2 > flag4 || flag4 > 3))) {};
    flag1 = 0;
  }
}

void* thr2(void * arg) {
  while (1) {
    flag2 = 1;
    while (!(flag1 < 3 && flag3 < 3 && flag4 < 3)) {};
    flag2 = 3;
    if (flag1 == 1 || flag3 == 1 || flag4 == 1) {
      flag2 = 2;
      while (!(flag1 == 4 || flag3 == 4 || flag4 == 4)) {};
    }
    flag2 = 4;
    while (!(flag1 < 2 && flag3 < 2 && flag4 < 2)) {};
    // begin critical section
    // end critical section
    while (!((2 > flag1 || flag1 > 3) && (2 > flag3 || flag3 > 3) && (2 > flag4 || flag4 > 3))) {};
    flag2 = 0;
  }
}

void* thr3(void * arg) {
  while (1) {
    flag3 = 1;
    while (!(flag1 < 3 && flag2 < 3 && flag4 < 3)) {};
    flag3 = 3;
    if (flag1 == 1 || flag2 == 1 || flag4 == 1) {
      flag3 = 2;
      while (!(flag1 == 4 || flag2 == 4 || flag4 == 4)) {};
    }
    flag3 = 4;
    while (!(flag1 < 2 && flag2 < 2 && flag4 < 2)) {};
    // begin critical section
    // end critical section
    while (!((2 > flag1 || flag1 > 3) && (2 > flag2 || flag2 > 3) && (2 > flag4 || flag4 > 3))) {};
    flag3 = 0;
  }
}

void* thr4(void * arg) {
  while (1) {
    flag4 = 1;
    while (!(flag1 < 3 && flag2 < 3 && flag3 < 3)) {};
    flag4 = 3;
    if (flag1 == 1 || flag2 == 1 || flag3 == 1) {
      flag4 = 2;
      while (!(flag1 == 4 || flag2 == 4 || flag3 == 4)) {};
    }
    flag4 = 4;
    while (!(flag1 < 2 && flag2 < 2 && flag3 < 2)) {};
    // begin critical section
    // end critical section
    while (!((2 > flag1 || flag1 > 3) && (2 > flag2 || flag2 > 3) && (2 > flag3 || flag3 > 3))) {};
    flag4 = 0;
  }
}

int main()
{
  __CPROVER_ASYNC_1: thr1(0);
  __CPROVER_ASYNC_2: thr2(0);
  __CPROVER_ASYNC_3: thr3(0);
  thr4(0);
}
