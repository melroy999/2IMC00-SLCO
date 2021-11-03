import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;
import java.util.Arrays;

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
            int i = start;
            Arrays.sort(lock_ids, start, end);
            for (; i < end; i++) {
                locks[lock_ids[i]].lock();
            }
        }

        // Unlock method
        void unlock(int[] lock_ids, int end) {
            for (int i = 0; i < end; i++) {
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
                random = new Random();
                lockManager = lockManagerInstance;
                lock_ids = new int[1];
                currentState = SM1Thread.States.SMC0;
            }

            // Representation of SLCO transition SMC0_0 (SMC0 -> SMC0)
            private boolean execute_transition_SMC0_0() {
                // SLCO statement: true -> true
                if (!(true)) {
                    return false;
                }

                // SLCO statement: x'[0] := i' -> x'[0] := i'
                x[0] = i;

                // SLCO statement: x'[i'] := 1 -> x'[i'] := 1
                x[i] = 1;

                // SLCO statement: x'[i' + 1] := 0 -> x'[i' + 1] := 0
                x[i + 1] = 0;

                // SLCO statement: [i' := 0; x'[y'[i']] := 1] -> [true; i' := 0; x'[y'[i']] := 1]
                if (!(true)) {
                    return false;
                }
                i = 0;
                x[y[i]] = 1;

                // SLCO statement: y'[z'[i'] + 1] := 0 -> y'[z'[i'] + 1] := 0
                y[z[i] + 1] = 0;

                // SLCO statement: z'[x'[i'] + 1] := 0 -> z'[x'[i'] + 1] := 0
                z[x[i] + 1] = 0;

                // SLCO statement: x'[x'[i']] := 0 -> x'[x'[i']] := 0
                x[x[i]] = 0;

                // SLCO statement: y'[i'] := 0 -> y'[i'] := 0
                y[i] = 0;

                return true;
            }

            // Representation of SLCO transition SMC0_1 (SMC0 -> SMC0)
            private boolean execute_transition_SMC0_1() {
                // SLCO statement: true -> true
                if (!(true)) {
                    return false;
                }

                // SLCO statement: x'[0] := i' -> x'[0] := i'
                x[0] = i;

                // SLCO statement: x'[i'] := 1 -> x'[i'] := 1
                x[i] = 1;

                // SLCO statement: x'[i' + 1] := 0 -> x'[i' + 1] := 0
                x[i + 1] = 0;

                // SLCO statement: [i' := 0; x'[y'[i']] := 1] -> [true; i' := 0; x'[y'[i']] := 1]
                if (!(true)) {
                    return false;
                }
                i = 0;
                x[y[i]] = 1;

                // SLCO statement: y'[z'[i'] + 1] := 0 -> y'[z'[i'] + 1] := 0
                y[z[i] + 1] = 0;

                // SLCO statement: z'[x'[i'] + 1] := 0 -> z'[x'[i'] + 1] := 0
                z[x[i] + 1] = 0;

                // SLCO statement: x'[x'[i']] := 0 -> x'[x'[i']] := 0
                x[x[i]] = 0;

                // SLCO statement: y'[i'] := 0 -> y'[i'] := 0
                y[i] = 0;

                return true;
            }

            private void exec_SMC0() {
                if (execute_transition_SMC0_0()) {
                    currentState = SM1Thread.States.SMC0;
                } else if (execute_transition_SMC0_1()) {
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