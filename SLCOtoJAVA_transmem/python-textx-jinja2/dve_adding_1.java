import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;
import java.util.Arrays;

// main class
public class dve_adding_1 {
  // The threads
  java_a1Thread java_T_a1;
  java_a2Thread java_T_a2;

  // Enum type for state machine states
  public enum java_State {
  R, Q, S
  }

  // Global variables
  public int c;
  public int x1;
  public int x2;

	class java_a1Thread extends Thread {
		private Thread java_t;
		private String java_threadName = "a1Thread";
		// Current state
		private dve_adding_1.java_State java_currentState;
		// Random number generator to handle non-determinism
		private Random java_randomGenerator;
		// Volatile variable to create memory fences
		private volatile boolean fence_var;
		// Thread local variables

		// Constructor
		java_a1Thread () {
			java_randomGenerator = new Random();
			java_currentState = dve_adding_1.java_State.Q;
		}

		// Execute method
		public void exec() {
			// variable to store non-deterministic choices
			int java_choice;
			while(true) {
				switch(java_currentState) {
					case Q:
						//System.out.println("a1_Q_<textx:slco2.Assignment instance at 0x2648bfacf98>");
						// c := 20
 
						c = 20;
 
						//System.out.println("a1_Q_<textx:slco2.MemoryFence instance at 0x2648bff4160>");
						// |#|
 
						fence_var = true;
 
						//System.out.println("a1_Q_<textx:slco2.Composite instance at 0x2648bff4c88>");
						// [c > 20; x1 := c]
 
						atomic {
							if (!(c > 20)) {
                                x1 = c;
								}
						}
 
						//System.out.println("a1_Q_<textx:slco2.Composite instance at 0x2648bff4080>");
						// [ c := 20; x1 := x2]
 
						atomic {
							c = 20;
                                x1 = x2;
						}
 
						// Change state
						java_currentState = dve_adding_1.java_State.R;
						break;
					case R:
						//System.out.println("a1_R_<textx:slco2.Composite instance at 0x2648bfea2e8>");
						// [ x1 := x1 + c]
 
						atomic {
							x1 = x1 + c;
						}
 
						// Change state
						java_currentState = dve_adding_1.java_State.S;
						break;
					case S:
						//System.out.println("a1_S_<textx:slco2.Composite instance at 0x2648bffc908>");
						// [ x1 := x1 + c]
 
						atomic {
							x1 = x1 + c;
						}
 
						// Change state
						java_currentState = dve_adding_1.java_State.Q;
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
	class java_a2Thread extends Thread {
		private Thread java_t;
		private String java_threadName = "a2Thread";
		// Current state
		private dve_adding_1.java_State java_currentState;
		// Random number generator to handle non-determinism
		private Random java_randomGenerator;
		// Volatile variable to create memory fences
		private volatile boolean fence_var;
		// Thread local variables

		// Constructor
		java_a2Thread () {
			java_randomGenerator = new Random();
			java_currentState = dve_adding_1.java_State.Q;
		}

		// Execute method
		public void exec() {
			// variable to store non-deterministic choices
			int java_choice;
			while(true) {
				switch(java_currentState) {
					case Q:
						//System.out.println("a2_Q_<textx:slco2.Composite instance at 0x2648bfe1208>");
						// [c < 20; x2 := c]
 
						atomic {
							if (!(c < 20)) {
                                x2 = c;
								}
						}
 
						// Change state
						java_currentState = dve_adding_1.java_State.R;
						break;
					case R:
						//System.out.println("a2_R_<textx:slco2.Composite instance at 0x2648c011438>");
						// [ x2 := x2 + c]
 
						atomic {
							x2 = x2 + c;
						}
 
						// Change state
						java_currentState = dve_adding_1.java_State.S;
						break;
					case S:
						//System.out.println("a2_S_<textx:slco2.Composite instance at 0x2648c0117b8>");
						// [ c := x2]
 
						atomic {
							c = x2;
						}
 
						// Change state
						java_currentState = dve_adding_1.java_State.Q;
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
	dve_adding_1() {
		// Instantiate global variables
		c = 1;
		x1 = 0;
		x2 = 0;
		java_T_a1 = new dve_adding_1.java_a1Thread();
		java_T_a2 = new dve_adding_1.java_a2Thread();
	}

	// Start all threads
	public void startThreads() {
		java_T_a1.start();
		java_T_a2.start();
	}

	// Join all threads
	public void joinThreads() {
		while (true) {
			try {
				java_T_a1.join();
				java_T_a2.join();
				break;
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	// Run application
	public static void main(String args[]) {
    dve_adding_1 java_ap = new dve_adding_1();
    java_ap.startThreads();
    java_ap.joinThreads();
	}
}