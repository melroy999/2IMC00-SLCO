import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;
import java.util.Arrays;

// main class
public class Test {
  // The threads
  java_SM4Thread java_T_SM4;

  // Enum type for state machine states
  public enum java_State {
  SMC0, SMC1
  }

  // Global variables
  public volatile int[] x;
  public volatile int i;

	// Lock class to handle locks of global variables
	class java_Keeper {
    // The locks
    ReentrantLock[] locks;
    // Which locks need to be acquired?
    boolean[] lockneeded;

		// Constructor
		java_Keeper() {
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

	class java_SM4Thread extends Thread {
		private Thread java_t;
		private String java_threadName = "SM4Thread";
		// Current state
		private Test.java_State java_currentState;
		// Random number generator to handle non-determinism
		private Random java_randomGenerator;
		// Keeper of global variables
		private Test.java_Keeper java_kp;
		// Array to store IDs of locks to be acquired
		private int[] java_lockIDs;
		// Thread local variables

		// Constructor
		java_SM4Thread (Test.java_Keeper java_k) {
			java_randomGenerator = new Random();
			java_currentState = Test.java_State.SMC0;
            java_kp = java_k;
            java_lockIDs = new int[2];
		}

		// Transition functions
        
        boolean execute_SMC0_0() {
          // [ i >= 0 && i < 1 && x[i] != 0; i := i + 1; i := i + 1 ]
          //System.out.println("SM4_SMC0_0");
          java_lockIDs[0] = 0;
          //System.out.println("SM4_SMC0_1");
          java_lockIDs[1] = 1 + i;
          //System.out.println("SM4_SMC0__sort");
          Arrays.sort(java_lockIDs,0,2);
          //System.out.println("SM4_SMC0__lock");
          java_kp.lock(java_lockIDs, 2);
          if (!(i >= 0 && i < 1 && x[i] != 0)) { java_kp.unlock(java_lockIDs, 2); return false; }
          i = i + 1;
          i = i + 1;
          //System.out.println("SM4_SMC0__unlock");
          java_kp.unlock(java_lockIDs, 2);
          return true;
        }

		// Execute method
		public void exec() {
			// variable to store non-deterministic choices
			int java_choice;
			while(true) {
				switch(java_currentState) {
					case SMC0:
                        if (execute_SMC0_0()) {
						  // Change state
						  java_currentState = Test.java_State.SMC0;
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
			if (java_t == null) {
				java_t = new Thread(this);
				java_t.start();
			}
		}
	}

	// Constructor for main class
	Test() {
		// Instantiate global variables
		x = new int[] {0,0};
		i = 0;
		Test.java_Keeper java_k = new Test.java_Keeper();
		java_T_SM4 = new Test.java_SM4Thread(java_k);
	}

	// Start all threads
	public void startThreads() {
		java_T_SM4.start();
	}

	// Join all threads
	public void joinThreads() {
		while (true) {
			try {
				java_T_SM4.join();
				break;
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	// Run application
	public static void main(String args[]) {
    Test java_ap = new Test();
    java_ap.startThreads();
    java_ap.joinThreads();
	}
}