
int flag1 = 0, flag2 = 0, flag3 = 0, flag4 = 0, flag5 = 0, flag6 = 0; // N integer flags

void* thr1(void * arg) {
  while (1) {
    flag1 = 1;
    __CPROVER_assume ((flag2 < 3 && flag3 < 3 && flag4 < 3 && flag5 < 3 && flag6 < 3));
    flag1 = 3;
    if (flag2 == 1 || flag3 == 1 || flag4 == 1 || flag5 == 1 || flag6 == 1) {
      flag1 = 2;
      __CPROVER_assume ((flag2 == 4 || flag3 == 4 || flag4 == 4 || flag5 == 4 || flag6 == 4));
    }
    flag1 = 4;
    __CPROVER_assume ((flag2 < 2 && flag3 < 2 && flag4 < 2 && flag5 < 2 && flag6 < 2));
    // begin critical section
    // end critical section
    __CPROVER_assume (((2 > flag2 || flag2 > 3) && (2 > flag3 || flag3 > 3) && (2 > flag4 || flag4 > 3) && (2 > flag5 || flag5 > 3) && (2 > flag6 || flag6 > 3)));
    flag1 = 0;
  }
}

void* thr2(void * arg) {
  while (1) {
    flag2 = 1;
    __CPROVER_assume ((flag1 < 3 && flag3 < 3 && flag4 < 3 && flag5 < 3 && flag6 < 3));
    flag2 = 3;
    if (flag1 == 1 || flag3 == 1 || flag4 == 1 || flag5 == 1 || flag6 == 1) {
      flag2 = 2;
      __CPROVER_assume ((flag1 == 4 || flag3 == 4 || flag4 == 4 || flag5 == 4 || flag6 == 4));
    }
    flag2 = 4;
    __CPROVER_assume ((flag1 < 2 && flag3 < 2 && flag4 < 2 && flag5 < 2 && flag6 < 2));
    // begin critical section
    // end critical section
    __CPROVER_assume (((2 > flag1 || flag1 > 3) && (2 > flag3 || flag3 > 3) && (2 > flag4 || flag4 > 3) && (2 > flag5 || flag5 > 3) && (2 > flag6 || flag6 > 3)));
    flag2 = 0;
  }
}

void* thr3(void * arg) {
  while (1) {
    flag3 = 1;
    __CPROVER_assume ((flag1 < 3 && flag2 < 3 && flag4 < 3 && flag5 < 3 && flag6 < 3));
    flag3 = 3;
    if (flag1 == 1 || flag2 == 1 || flag4 == 1 || flag5 == 1 || flag6 == 1) {
      flag3 = 2;
      __CPROVER_assume ((flag1 == 4 || flag2 == 4 || flag4 == 4 || flag5 == 4 || flag6 == 4));
    }
    flag3 = 4;
    __CPROVER_assume ((flag1 < 2 && flag2 < 2 && flag4 < 2 && flag5 < 2 && flag6 < 2));
    // begin critical section
    // end critical section
    __CPROVER_assume (((2 > flag1 || flag1 > 3) && (2 > flag2 || flag2 > 3) && (2 > flag4 || flag4 > 3) && (2 > flag5 || flag5 > 3) && (2 > flag6 || flag6 > 3)));
    flag3 = 0;
  }
}

void* thr4(void * arg) {
  while (1) {
    flag4 = 1;
    __CPROVER_assume ((flag1 < 3 && flag2 < 3 && flag3 < 3 && flag5 < 3 && flag6 < 3));
    flag4 = 3;
    if (flag1 == 1 || flag2 == 1 || flag3 == 1 || flag5 == 1 || flag6 == 1) {
      flag4 = 2;
      __CPROVER_assume ((flag1 == 4 || flag2 == 4 || flag3 == 4 || flag5 == 4 || flag6 == 4));
    }
    flag4 = 4;
    __CPROVER_assume ((flag1 < 2 && flag2 < 2 && flag3 < 2 && flag5 < 2 && flag6 < 2));
    // begin critical section
    // end critical section
    __CPROVER_assume (((2 > flag1 || flag1 > 3) && (2 > flag2 || flag2 > 3) && (2 > flag3 || flag3 > 3) && (2 > flag5 || flag5 > 3) && (2 > flag6 || flag6 > 3)));
    flag4 = 0;
  }
}

void* thr5(void * arg) {
  while (1) {
    flag5 = 1;
    __CPROVER_assume ((flag1 < 3 && flag2 < 3 && flag3 < 3 && flag4 < 3 && flag6 < 3));
    flag5 = 3;
    if (flag1 == 1 || flag2 == 1 || flag3 == 1 || flag4 == 1 || flag6 == 1) {
      flag5 = 2;
      __CPROVER_assume ((flag1 == 4 || flag2 == 4 || flag3 == 4 || flag4 == 4 || flag6 == 4));
    }
    flag5 = 4;
    __CPROVER_assume ((flag1 < 2 && flag2 < 2 && flag3 < 2 && flag4 < 2 && flag6 < 2));
    // begin critical section
    // end critical section
    __CPROVER_assume (((2 > flag1 || flag1 > 3) && (2 > flag2 || flag2 > 3) && (2 > flag3 || flag3 > 3) && (2 > flag4 || flag4 > 3) && (2 > flag6 || flag6 > 3)));
    flag5 = 0;
  }
}

void* thr6(void * arg) {
  while (1) {
    flag6 = 1;
    __CPROVER_assume ((flag1 < 3 && flag2 < 3 && flag3 < 3 && flag4 < 3 && flag5 < 3));
    flag6 = 3;
    if (flag1 == 1 || flag2 == 1 || flag3 == 1 || flag4 == 1 || flag5 == 1) {
      flag6 = 2;
      __CPROVER_assume ((flag1 == 4 || flag2 == 4 || flag3 == 4 || flag4 == 4 || flag5 == 4));
    }
    flag6 = 4;
    __CPROVER_assume ((flag1 < 2 && flag2 < 2 && flag3 < 2 && flag4 < 2 && flag5 < 2));
    // begin critical section
    // end critical section
    __CPROVER_assume (((2 > flag1 || flag1 > 3) && (2 > flag2 || flag2 > 3) && (2 > flag3 || flag3 > 3) && (2 > flag4 || flag4 > 3) && (2 > flag5 || flag5 > 3)));
    flag6 = 0;
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
