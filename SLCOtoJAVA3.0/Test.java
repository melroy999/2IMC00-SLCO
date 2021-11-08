import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

// Main class
public class Test {
    // The objects in the model
    private final SLCO_Class[] objects;

    // Lock class to handle locks of global variables
    private static class LockManager {
        // The locks
        private final ReentrantLock[] locks;

        LockManager(int noVariables) {
            locks = new ReentrantLock[noVariables];
            for(int i = 0; i < locks.length; i++) {
                locks[i] = new ReentrantLock();
            }
        }

        // Lock method
        void lock(int[] lock_ids, int start, int end) {
            Arrays.sort(lock_ids, start, end);
            for (int i = start; i < end; i++) {
                locks[lock_ids[i]].lock();
            }
        }

        // Unlock method
        void unlock(int[] lock_ids, int start, int end) {
            for (int i = start; i < end; i++) {
                locks[lock_ids[i]].unlock();
            }
        }

        // Unlock method during exceptions
        void exception_unlock() {
            System.err.println("Exception encountered. Releasing all locks currently owned by " + Thread.currentThread().getName() + ".");
            for(ReentrantLock lock: locks) {
                while(lock.isHeldByCurrentThread()) {
                    lock.unlock();
                }
            }
        }
    }

    // Template for SLCO classes
    interface SLCO_Class {
        void startThreads();
        void joinThreads();
    }

    // Representation of the SLCO class P
    private static class P implements SLCO_Class {
        // The threads
        private final Thread T_SM1;

        // Global variables
        private volatile int[] x;
        private volatile int[] y;
        private volatile int[] z;
        private volatile int i;

        // Define the states fot the state machine SM1
        interface P_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1
            }
        }

        // Representation of the SLCO state machine SM1
        class SM1Thread extends Thread implements P_SM1Thread_States {
            // Current state
            private SM1Thread.States currentState;

            // Random number generator to handle non-determinism
            private final Random random;

            // The lock manager
            private final LockManager lockManager;

            // A list of lock ids that can be reused
            private final int[] lock_ids;

            SM1Thread (LockManager lockManagerInstance) {
                currentState = SM1Thread.States.SMC0;
                lockManager = lockManagerInstance;
                lock_ids = new int[5];
                random = new Random();
            }

            // Representation of SLCO transition SMC0_0 (SMC0 -> SMC0)
            private boolean execute_transition_SMC0_0() {
                // SLCO statement: true -> true
                if (!(true)) {
                    return false;
                }

                // SLCO statement: x[0] := i -> x[0] := i
                // P1
                lock_ids[0] = 0; // Acquire i
                lock_ids[1] = 3 + 0; // Acquire x[0]
                lockManager.lock(lock_ids, 0, 1);
                x[0] = i;
                // P4
                lockManager.unlock(lock_ids, 0, 1); // Release i, x[0]

                // SLCO statement: x[i] := 1 -> x[i] := 1
                // P1
                lock_ids[0] = 0; // Acquire i
                lock_ids[1] = 3 + i; // Acquire x[i]
                lockManager.lock(lock_ids, 0, 1);
                x[i] = 1;
                // P4
                lockManager.unlock(lock_ids, 0, 1); // Release i, x[i]

                // SLCO statement: x[i + 1] := 0 -> x[i + 1] := 0
                // P1
                lock_ids[0] = 0; // Acquire i
                lock_ids[1] = 3 + i + 1; // Acquire x[i + 1]
                lockManager.lock(lock_ids, 0, 1);
                x[i + 1] = 0;
                // P4
                lockManager.unlock(lock_ids, 0, 1); // Release i, x[i + 1]

                // SLCO statement: [i := 0; x[y[i]] := 1] -> [true; i := 0; x[y[i]] := 1]
                if (!(true)) {
                    return false;
                }
                i = 0;
                x[y[i]] = 1;

                // SLCO statement: y[z[i] + 1] := 0 -> y[z[i] + 1] := 0
                // P1
                lock_ids[0] = 0; // Acquire i
                lock_ids[1] = 1 + 1; // Acquire y[1]
                lock_ids[2] = 1 + 0; // Acquire y[0]
                lock_ids[3] = 5 + i; // Acquire z[i]
                lockManager.lock(lock_ids, 0, 3);
                // P2
                lock_ids[4] = 1 + z[i] + 1; // Acquire y[z[i] + 1]
                lockManager.lock(lock_ids, 4, 4);
                // P3
                lockManager.unlock(lock_ids, 1, 2); // Release y[1], y[0]
                y[z[i] + 1] = 0;
                // P4
                lockManager.unlock(lock_ids, 0, 0); // Release i
                lockManager.unlock(lock_ids, 3, 4); // Release z[i], y[z[i] + 1]

                // SLCO statement: z[x[i] + 1] := 0 -> z[x[i] + 1] := 0
                // P1
                lock_ids[0] = 0; // Acquire i
                lock_ids[1] = 3 + i; // Acquire x[i]
                lock_ids[2] = 5 + x[i] + 1; // Acquire z[x[i] + 1]
                lockManager.lock(lock_ids, 0, 2);
                z[x[i] + 1] = 0;
                // P4
                lockManager.unlock(lock_ids, 0, 2); // Release i, x[i], z[x[i] + 1]

                // SLCO statement: x[x[i]] := 0 -> x[x[i]] := 0
                // P1
                lock_ids[0] = 0; // Acquire i
                lock_ids[1] = 3 + 0; // Acquire x[0]
                lock_ids[2] = 3 + 1; // Acquire x[1]
                lockManager.lock(lock_ids, 0, 2);
                // P2
                lock_ids[3] = 3 + i; // Acquire x[i]
                lock_ids[4] = 3 + x[i]; // Acquire x[x[i]]
                lockManager.lock(lock_ids, 3, 4);
                // P3
                lockManager.unlock(lock_ids, 1, 2); // Release x[0], x[1]
                x[x[i]] = 0;
                // P4
                lockManager.unlock(lock_ids, 0, 0); // Release i
                lockManager.unlock(lock_ids, 3, 4); // Release x[i], x[x[i]]

                // SLCO statement: x[x[i]] = 1 -> x[x[i]] = 1
                // P1
                lock_ids[0] = 0; // Acquire i
                lock_ids[1] = 3 + 0; // Acquire x[0]
                lock_ids[2] = 3 + 1; // Acquire x[1]
                lockManager.lock(lock_ids, 0, 2);
                // P2
                lock_ids[3] = 3 + i; // Acquire x[i]
                lock_ids[4] = 3 + x[i]; // Acquire x[x[i]]
                lockManager.lock(lock_ids, 3, 4);
                // P3
                lockManager.unlock(lock_ids, 1, 2); // Release x[0], x[1]
                try {
                    if (!(x[x[i]] == 1)) {
                        return false;
                    }
                } finally {
                    // P4
                    lockManager.unlock(lock_ids, 0, 0); // Release i
                    lockManager.unlock(lock_ids, 3, 4); // Release x[i], x[x[i]]
                }

                // SLCO statement: y[i] := 0 -> y[i] := 0
                // P1
                lock_ids[0] = 0; // Acquire i
                lock_ids[1] = 1 + i; // Acquire y[i]
                lockManager.lock(lock_ids, 0, 1);
                y[i] = 0;
                // P4
                lockManager.unlock(lock_ids, 0, 1); // Release i, y[i]

                return true;
            }

            private void exec_SMC0() {
                if (execute_transition_SMC0_0()) {
                    currentState = SM1Thread.States.SMC0;
                }
            }

            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            // Execute method
            private void exec() {
                while(true) {
                    switch(currentState) {
                        case SMC0 -> exec_SMC0();
                        case SMC1 -> exec_SMC1();
                    }
                }
            }

            // Run method
            public void run() {
                exec();
            }
        }

        P(int[] x, int[] y, int[] z, int i) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(1);

            // Instantiate global variables
            this.x = x;
            this.y = y;
            this.z = z;
            this.i = i;

            // Instantiate state machines
            T_SM1 = new SM1Thread(lockManager);
        }

        // Start all threads
        public void startThreads() {
            T_SM1.start();
        }

        // Join all threads
        public void joinThreads() {
            while (true) {
                try {
                    T_SM1.join();
                    break;
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    Test() {
        // Instantiate the objects
        objects = new SLCO_Class[] {
            new P(
                new int[]{0, 0},
                new int[]{0, 0},
                new int[]{0, 0},
                2
            )
        };
    }

    // Start all threads
    private void startThreads() {
        for(SLCO_Class object : objects) {
            object.startThreads();
        }
    }

    // Join all threads
    private void joinThreads() {
        for(SLCO_Class object : objects) {
            object.joinThreads();
        }
    }

    // Run application
    public static void main(String[] args) {
        Test model = new Test();
        model.startThreads();
        model.joinThreads();
    }
}