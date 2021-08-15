// #include <assert.h>

int choosing[4];
int number[4];

void* P0(void * arg) {
    int j;
    int max;
    int tmp, tmp2;

    // NCS
    while (1) {
      choosing[0] = 1;
      j = 0;
      max = 0;
      // choose
      while (1) {
        if (j < 4) {
            // choose2
            if (number[j] > max) {
                max = number[j];
                j = j+1;
                continue;
                // -> choose
            }
            else {
                j = j + 1;
                continue;
                // -> choose
            }
        }
        else if (j == 4 && max < 7) {
            number[0] = max+1;
            j = 0;
            choosing[0] = 0;
            // -> forloop
            while (1) {
                if (j < 4 && choosing[j] == 0) {
                    // wait
                    tmp = number[j];
                    tmp2 = number[0];
                    if (tmp == 0 || (tmp > tmp2) || (tmp == tmp2 && 0 <= j)) {
                        j = j+1;
                        continue;
                        // -> forloop
                    }
                }
                else if (j == 4) {
                    // CS
                    number[0] = 0;
                    break;
                    // -> NCS
                }
            }
            break;
            // -> NCS
        }
      }
		}
}

void* P1(void * arg) {
    int j;
    int max;
    int tmp, tmp2;

    // NCS
    while (1) {
      choosing[1] = 1;
      j = 0;
      max = 0;
      // choose
      while (1) {
        if (j < 4) {
            // choose2
            if (number[j] > max) {
                max = number[j];
                j = j+1;
                continue;
                // -> choose
            }
            else {
                j = j + 1;
                continue;
                // -> choose
            }
        }
        else if (j == 4 && max < 7) {
            number[1] = max+1;
            j = 0;
            choosing[1] = 0;
            // -> forloop
            while (1) {
                if (j < 4 && choosing[j] == 0) {
                    // wait
                    tmp = number[j];
                    tmp2 = number[1];
                    if (tmp == 0 || (tmp > tmp2) || (tmp == tmp2 && 0 <= j)) {
                        j = j+1;
                        continue;
                        // -> forloop
                    }
                }
                else if (j == 4) {
                    // CS
                    number[1] = 0;
                    break;
                    // -> NCS
                }
            }
            break;
            // -> NCS
        }
      }
		}
}

void* P2(void * arg) {
    int j;
    int max;
    int tmp, tmp2;

    // NCS
    while (1) {
      choosing[2] = 1;
      j = 0;
      max = 0;
      // choose
      while (1) {
        if (j < 4) {
            // choose2
            if (number[j] > max) {
                max = number[j];
                j = j+1;
                continue;
                // -> choose
            }
            else {
                j = j + 1;
                continue;
                // -> choose
            }
        }
        else if (j == 4 && max < 7) {
            number[2] = max+1;
            j = 0;
            choosing[2] = 0;
            // -> forloop
            while (1) {
                if (j < 4 && choosing[j] == 0) {
                    // wait
                    tmp = number[j];
                    tmp2 = number[2];
                    if (tmp == 0 || (tmp > tmp2) || (tmp == tmp2 && 0 <= j)) {
                        j = j+1;
                        continue;
                        // -> forloop
                    }
                }
                else if (j == 4) {
                    // CS
                    number[2] = 0;
                    break;
                    // -> NCS
                }
            }
            break;
            // -> NCS
        }
      }
	 }
}

void* P3(void * arg) {
    int j;
    int max;
    int tmp, tmp2;

    // NCS
    while (1) {
      choosing[3] = 1;
      j = 0;
      max = 0;
      // choose
      while (1) {
        if (j < 4) {
            // choose2
            if (number[j] > max) {
                max = number[j];
                j = j+1;
                continue;
                // -> choose
            }
            else {
                j = j + 1;
                continue;
                // -> choose
            }
        }
        else if (j == 4 && max < 7) {
            number[3] = max+1;
            j = 0;
            choosing[3] = 0;
            // -> forloop
            while (1) {
                if (j < 4 && choosing[j] == 0) {
                    // wait
                    tmp = number[j];
                    tmp2 = number[3];
                    if (tmp == 0 || (tmp > tmp2) || (tmp == tmp2 && 0 <= j)) {
                        j = j+1;
                        continue;
                        // -> forloop
                    }
                }
                else if (j == 4) {
                    // CS
                    number[3] = 0;
                    break;
                    // -> NCS
                }
            }
            break;
            // -> NCS
        }
     }
	}
}

int main()
{
  __CPROVER_ASYNC_1: P0(0);
  __CPROVER_ASYNC_2: P1(0);
  __CPROVER_ASYNC_3: P2(0);
  P3(0);
}
