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
        private volatile boolean[] b;
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
                lock_ids = new int[0];
                random = new Random();
            }

            // Representation of SLCO transition SMC0_0 (SMC0 -> SMC0)
            private boolean execute_transition_SMC0_0() {
                // SLCO statement: [x[0] > 0; i := i + 1; x[i] := 1] -> [x[0] > 0; i := i + 1; x[i] := 1]
                // SLCO statement: x[0] > 0 -> x[0] > 0
                if (!(x[0] > 0)) {
                    return false;
                }
                // SLCO statement: i := i + 1 -> i := i + 1
                i = i + 1;
                // SLCO statement: x[i] := 1 -> x[i] := 1
                x[i] = 1;

                // SLCO statement: [x[0] > 0; i := 2; x[i] := 1; x[1] := 1] -> [x[0] > 0; i := 2; x[i] := 1; x[1] := 1]
                // SLCO statement: x[0] > 0 -> x[0] > 0
                if (!(x[0] > 0)) {
                    return false;
                }
                // SLCO statement: i := 2 -> i := 2
                i = 2;
                // SLCO statement: x[i] := 1 -> x[i] := 1
                x[i] = 1;
                // SLCO statement: x[1] := 1 -> x[1] := 1
                x[1] = 1;

                // SLCO statement: [x[0] > 0; i := 0; x[i] := 1; x[1] := 1] -> [x[0] > 0; i := 0; x[i] := 1; x[1] := 1]
                // SLCO statement: x[0] > 0 -> x[0] > 0
                if (!(x[0] > 0)) {
                    return false;
                }
                // SLCO statement: i := 0 -> i := 0
                i = 0;
                // SLCO statement: x[i] := 1 -> x[i] := 1
                x[i] = 1;
                // SLCO statement: x[1] := 1 -> x[1] := 1
                x[1] = 1;

                // SLCO statement: [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1] -> [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1]
                // SLCO statement: i >= 0 and i < 3 and b[i] -> i >= 0 and i < 3 and b[i]
                if (!(i >= 0 && i < 3 && b[i])) {
                    return false;
                }
                // SLCO statement: i := 0 -> i := 0
                i = 0;
                // SLCO statement: x[i] := 1 -> x[i] := 1
                x[i] = 1;
                // SLCO statement: x[1] := 1 -> x[1] := 1
                x[1] = 1;

                // SLCO statement: [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0] -> [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0]
                // SLCO statement: i >= 0 and i < 2 -> i >= 0 and i < 2
                if (!(i >= 0 && i < 2)) {
                    return false;
                }
                // SLCO statement: i := 0 -> i := 0
                i = 0;
                // SLCO statement: x[i] := 0 -> x[i] := 0
                x[i] = 0;
                // SLCO statement: x[x[i]] := 0 -> x[x[i]] := 0
                x[x[i]] = 0;

                // SLCO statement: [i := 0; b[i] := i >= 0 and i < 2 and b[i]] -> [true; i := 0; b[i] := i >= 0 and i < 2 and b[i]]
                // SLCO statement: true -> true
                if (!(true)) {
                    return false;
                }
                // SLCO statement: i := 0 -> i := 0
                i = 0;
                // SLCO statement: b[i] := i >= 0 and i < 2 and b[i] -> b[i] := i >= 0 and i < 2 and b[i]
                b[i] = i >= 0 && i < 2 && b[i];

                // SLCO statement: [b[0] := 0; i := 0; b[i] := i >= 0 and i < 2 and b[i]] -> [true; b[0] := 0; i := 0; b[i] := i >= 0 and i < 2 and b[i]]
                // SLCO statement: true -> true
                if (!(true)) {
                    return false;
                }
                // SLCO statement: b[0] := 0 -> b[0] := 0
                b[0] = 0;
                // SLCO statement: i := 0 -> i := 0
                i = 0;
                // SLCO statement: b[i] := i >= 0 and i < 2 and b[i] -> b[i] := i >= 0 and i < 2 and b[i]
                b[i] = i >= 0 && i < 2 && b[i];

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

        P(int[] x, int[] y, boolean[] b, int i) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(1);

            // Instantiate global variables
            this.x = x;
            this.y = y;
            this.b = b;
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
                new boolean[]{False, False, False},
                0
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