// #include <assert.h>

int choosing[2];
int number[2];

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
        if (j < 2) {
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
        else if (j == 2 && max < 9) {
            number[0] = max+1;
            j = 0;
            choosing[0] = 0;
            // -> forloop
            while (1) {
                if (j < 2 && choosing[j] == 0) {
                    // wait
                    tmp = number[j];
                    tmp2 = number[0];
                    __CPROVER_assume ((tmp == 0 || (tmp > tmp2) || (tmp == tmp2 && 0 <= j)));
                    j = j+1;
                    continue;
                    // -> forloop
                }
                else if (j == 2) {
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
        if (j < 2) {
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
        else if (j == 2 && max < 9) {
            number[1] = max+1;
            j = 0;
            choosing[1] = 0;
            // -> forloop
            while (1) {
                if (j < 2 && choosing[j] == 0) {
                    // wait
                    tmp = number[j];
                    tmp2 = number[1];
                    __CPROVER_assume ((tmp == 0 || (tmp > tmp2) || (tmp == tmp2 && 1 <= j)));
                    j = j+1;
                    continue;
                    // -> forloop
                }
                else if (j == 2) {
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

int main()
{
  __CPROVER_ASYNC_1: P0(0);
  P1(0);
}
