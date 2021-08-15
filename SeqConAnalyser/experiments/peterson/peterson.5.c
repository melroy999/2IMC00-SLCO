int turn; // integer variable to hold the ID of the thread whose turn is it
int flag1 = 0, flag2 = 0, flag3 = 0, flag4 = 0, flag5 = 0, flag6 = 0; // boolean flags

void* thr1(void * arg) { // frontend produces 12 transitions from this thread. It would be better if it would produce only 8!
  flag1 = 1;
  turn = 1;
  do {} while ((flag2==1 && turn==1) || (flag3==1 && turn==2) || (flag4==1 && turn==3) || (flag5==1 && turn==4) || (flag6==1 && turn==5));
  // begin: critical section
  // end: critical section
  flag1 = 0;
}

void* thr2(void * arg) {
  flag2 = 1;
  turn = 2;
  do {} while ((flag1==1 && turn==0) || (flag3==1 && turn==2) || (flag4==1 && turn==3) || (flag5==1 && turn==4) || (flag6==1 && turn==5));
  // begin: critical section
  // end: critical section
  flag2 = 0;
}

void* thr3(void * arg) {
  flag3 = 1;
  turn = 3;
  do {} while ((flag1==1 && turn==0) || (flag2==1 && turn==1) || (flag4==1 && turn==3) || (flag5==1 && turn==4) || (flag6==1 && turn==5));
  // begin: critical section
  // end: critical section
  flag3 = 0;
}

void* thr4(void * arg) {
  flag4 = 1;
  turn = 0;
  do {} while ((flag1==1 && turn==0) || (flag2==1 && turn==1) || (flag3==1 && turn==2) || (flag5==1 && turn==4) || (flag6==1 && turn==5));
  // begin: critical section
  // end: critical section
  flag4 = 0;
}

void* thr5(void * arg) {
  flag5 = 1;
  turn = 0;
  do {} while ((flag1==1 && turn==0) || (flag2==1 && turn==1) || (flag3==1 && turn==2) || (flag4==1 && turn==3) || (flag6==1 && turn==5));
  // begin: critical section
  // end: critical section
  flag5 = 0;
}

void* thr6(void * arg) {
  flag6 = 1;
  turn = 0;
  do {} while ((flag1==1 && turn==0) || (flag2==1 && turn==1) || (flag3==1 && turn==2) || (flag4==1 && turn==3) || (flag5==1 && turn==4));
  // begin: critical section
  // end: critical section
  flag6 = 0;
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
