import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;
import java.util.Arrays;

// main class
public class Test {
  // The threads
  SMCThread T_SMC;
  ComThread T_Com;

  // Upperbound for transition counter
  public static final long COUNTER_BOUND = 999999999L;

  // Enum type for state machine states
  public enum State {
  SMC1, SMC0, Com1, Com2, Com0
  }

  // Global variables
  public boolean[] x;
  public int y;

	// Lock class to handle locks of global variables
	class Keeper {
    // The locks
		ReentrantLock[] locks;
    // Which locks need to be acquired?
    boolean[] lockneeded;

		// Constructor
		Keeper() {
			locks = new ReentrantLock[3];
			lockneeded = new boolean[] { true,true,true };
			for (int i = 0; i < 3; i++) {
				locks[i] = new ReentrantLock(true);
			}
		}

		// Lock method
		public void lock(int[] l, int size) {
			for (int i = 0; i < size; i++) {
				if (lockneeded[l[i]]) {
          locks[l[i]].lock();
        }
      }
		}

		// Unlock method
		public void unlock(int[] l, int size) {
		  for (int i = 0; i < size; i++) {
			  if (lockneeded[l[i]]) {
          locks[l[i]].unlock();
        }
      }
		}
	}

	class SMCThread extends Thread {
		private Thread t;
		private String threadName = "SMCThread";
		// Current state
		private Test.State currentState;
		// Random number generator to handle non-determinism
		private Random randomGenerator;
		// Counter of main while-loop iterations
		long transcounter;
		// Keeper of global variables
		private Test.Keeper kp;
		// Array to store IDs of locks to be acquired
		private int[] lockIDs;
		// Thread local variables
		private int i;

		// Constructor
		SMCThread (Test.Keeper k) {
			randomGenerator = new Random();
			transcounter = 0;
			currentState = Test.State.SMC0;
      kp = k;
      lockIDs = new int[3];
			i = 0;
		}

		// Execute method
		public synchronized void exec() {
			// variable to store non-deterministic choices
			int choice;
			while(transcounter < COUNTER_BOUND) {
				switch(currentState) {
					case SMC0:
						//System.out.println("SMC_SMC0 " + transcounter);
						choice = randomGenerator.nextInt(2);
						switch(choice) {
							case 0:
								//System.out.println("SMC_SMC0_0");
								// [not x[i]; i := i + 1; x[i] := i == 2; i := 3; x[0] := False]
								//System.out.println("SMC_SMC0_0");
							  lockIDs[0] = 0 + (i + 1);
								//System.out.println("SMC_SMC0_1");
							  lockIDs[1] = 0 + 0;
								//System.out.println("SMC_SMC0_2");
							  lockIDs[2] = 0 + i;
								//System.out.println("SMC_SMC0_0_sort");
								Arrays.sort(lockIDs,0,3);
								//System.out.println("SMC_SMC0_0_lock");
								kp.lock(lockIDs, 3);
								//System.out.println("SMC_SMC0_0_stat");
								if (!(!(x[i]))) { kp.unlock(lockIDs, 3); transcounter++; break; }
                i = i + 1;
                x[i] = i == 2;
                i = 3;
                x[0] = false;
								//System.out.println("SMC_SMC0_0_unlock");
								kp.unlock(lockIDs, 3);
                notifyAll();
								// Change state
								//System.out.println("SMC_SMC0_0_changestate");
								currentState = Test.State.SMC1;
                // Increment counter
								//System.out.println("SMC_SMC0_0_increment");
                transcounter++;
								break;
							case 1:
								//System.out.println("SMC_SMC0_1");
								// i := 0
								//System.out.println("SMC_SMC0_1_sort");
								Arrays.sort(lockIDs,0,0);
								//System.out.println("SMC_SMC0_1_lock");
								kp.lock(lockIDs, 0);
								//System.out.println("SMC_SMC0_1_stat");
								i = 0;
								//System.out.println("SMC_SMC0_1_unlock");
								kp.unlock(lockIDs, 0);
                notifyAll();
								// Change state
								//System.out.println("SMC_SMC0_1_changestate");
								currentState = Test.State.SMC0;
                // Increment counter
								//System.out.println("SMC_SMC0_1_increment");
                transcounter++;
								break;
						}
						break;
					default:
						return;
				}
			}
		}

		// Run method
		public void run() {
			exec();
		}

		// Start method
		public void start() {
			if (t == null) {
				t = new Thread(this);
				t.start();
			}
		}
	}
	class ComThread extends Thread {
		private Thread t;
		private String threadName = "ComThread";
		// Current state
		private Test.State currentState;
		// Random number generator to handle non-determinism
		private Random randomGenerator;
		// Counter of main while-loop iterations
		long transcounter;
		// Keeper of global variables
		private Test.Keeper kp;
		// Array to store IDs of locks to be acquired
		private int[] lockIDs;
		// Thread local variables
		private int lx;

		// Constructor
		ComThread (Test.Keeper k) {
			randomGenerator = new Random();
			transcounter = 0;
			currentState = Test.State.Com0;
      kp = k;
      lockIDs = new int[0];
			lx = 0;
		}

		// Execute method
		public synchronized void exec() {
			// variable to store non-deterministic choices
			int choice;
			while(transcounter < COUNTER_BOUND) {
				switch(currentState) {
					case Com0:
						//System.out.println("Com_Com0 " + transcounter);
						// lx == 0
						Arrays.sort(lockIDs,0,0);
						kp.lock(lockIDs, 0);
						if (!(lx == 0)) { kp.unlock(lockIDs, 0); transcounter++; break; }
						kp.unlock(lockIDs, 0);
						// Change state
						currentState = Test.State.Com1;
            // Increment counter
            transcounter++;
						break;
					default:
						return;
				}
			}
		}

		// Run method
		public void run() {
			exec();
		}

		// Start method
		public void start() {
			if (t == null) {
				t = new Thread(this);
				t.start();
			}
		}
	}

  // Constructor for main class
  Test() {
    // Instantiate global variables
    x = new boolean[] {false,true};
    y = 0;
	  Test.Keeper k = new Test.Keeper();
		T_SMC = new Test.SMCThread(k);
		T_Com = new Test.ComThread(k);
  }

  // Start all threads
  public void startThreads() {
    T_SMC.start();
    T_Com.start();
  }

  // Join all threads
  public void joinThreads() {
    while (true) {
      try {
			  T_SMC.join();
			  T_Com.join();
        break;
			} catch (InterruptedException e) {
        e.printStackTrace();
      }
		}
  }

	// Run application
	public static void main(String args[]) {
    Test ap = new Test();
    ap.startThreads();
    ap.joinThreads();
	}
}