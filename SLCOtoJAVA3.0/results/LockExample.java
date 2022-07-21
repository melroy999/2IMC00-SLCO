package testing;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.time.Duration;
import java.time.Instant;

// SLCO model LockExample.
public class LockExample {
    // The objects in the model.
    private final SLCO_Class[] objects;

    // Interface for SLCO classes.
    interface SLCO_Class {
        void startThreads();
        void joinThreads();
    }

    LockExample() {
        // Instantiate the objects.
        objects = new SLCO_Class[] {
            new P(
                0,
                new boolean[]{ false, false, false },
                new int[]{ 0, 0, 0 }
            )
        };
    }

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
        void acquire_locks(int[] lock_ids, int end) {
            Arrays.sort(lock_ids, 0, end);
            for (int i = 0; i < end; i++) {
                locks[lock_ids[i]].lock();
            }
        }

        // Unlock method
        void release_locks(int[] lock_ids, int end) {
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

    // Representation of the SLCO class P.
    private static class P implements SLCO_Class {
        // The state machine threads.
        private final Thread T_SM1;

        // Class variables.
        private volatile int i;
        private final boolean[] x;
        private final int[] y;

        P(int i, boolean[] x, int[] y) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(7);

            // Instantiate the class variables.
            this.i = i;
            this.x = x;
            this.y = y;

            // Instantiate the state machine threads and pass on the class' lock manager.
            T_SM1 = new P_SM1Thread(lockManager);
        }

        // Define the states fot the state machine SM1.
        interface P_SM1Thread_States {
            enum States {
                SMC0
            }
        }

        // Representation of the SLCO state machine SM1.
        class P_SM1Thread extends Thread implements P_SM1Thread_States {
            // Current state
            private P_SM1Thread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            P_SM1Thread(LockManager lockManagerInstance) {
                currentState = P_SM1Thread.States.SMC0;
                lockManager = lockManagerInstance;
                lock_ids = new int[5];
                target_locks = new int[9];
                random = new Random();
            }

            // SLCO expression wrapper | i >= 0.
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                return i >= 0;
            }

            // SLCO expression wrapper | i >= 0 and i < 3.
            private boolean t_SMC0_0_s_0_n_1() {
                if(t_SMC0_0_s_0_n_0() && i < 3) {
                    return true;
                }
                lock_ids[0] = target_locks[1] = 1 + 1; // Acquire x[1]
                lock_ids[1] = target_locks[2] = 1 + 0; // Acquire x[0]
                lock_ids[2] = target_locks[3] = 1 + 2; // Acquire x[2]
                lockManager.acquire_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | x[i].
            private boolean t_SMC0_0_s_0_n_2() {
                lock_ids[0] = target_locks[1] = 1 + 1; // Acquire x[1]
                lock_ids[1] = target_locks[2] = 1 + 0; // Acquire x[0]
                lock_ids[2] = target_locks[3] = 1 + 2; // Acquire x[2]
                lock_ids[3] = target_locks[4] = 1 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 4);
                if(x[i]) {
                    lock_ids[0] = target_locks[0]; // Release i
                    lock_ids[1] = target_locks[1]; // Release x[1]
                    lock_ids[2] = target_locks[2]; // Release x[0]
                    lock_ids[3] = target_locks[3]; // Release x[2]
                    lock_ids[4] = target_locks[4]; // Release x[i]
                    lockManager.release_locks(lock_ids, 5);
                    return true;
                }
                lock_ids[0] = target_locks[4]; // Release x[i]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 3 and x[i].
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | i >= 0 and i < 3 and x[i].
                if(!(t_SMC0_0_s_0_n_1() && t_SMC0_0_s_0_n_2())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | x[0].
            private boolean t_SMC0_1_s_0_n_0() {
                if(x[0]) {
                    lock_ids[0] = target_locks[0]; // Release i
                    lock_ids[1] = target_locks[1]; // Release x[1]
                    lock_ids[2] = target_locks[2]; // Release x[0]
                    lock_ids[3] = target_locks[3]; // Release x[2]
                    lockManager.release_locks(lock_ids, 4);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release x[0]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | x[1].
            private boolean t_SMC0_1_s_0_n_1() {
                if(x[1]) {
                    lock_ids[0] = target_locks[0]; // Release i
                    lock_ids[1] = target_locks[1]; // Release x[1]
                    lock_ids[2] = target_locks[3]; // Release x[2]
                    lockManager.release_locks(lock_ids, 3);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release x[1]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | x[2].
            private boolean t_SMC0_1_s_0_n_2() {
                if(x[2]) {
                    lock_ids[0] = target_locks[0]; // Release i
                    lock_ids[1] = target_locks[3]; // Release x[2]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[3]; // Release x[2]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | x[0] or x[1] or x[2].
            private boolean execute_transition_SMC0_1() {
                // SLCO expression | x[0] or x[1] or x[2].
                if(!(t_SMC0_1_s_0_n_0() || t_SMC0_1_s_0_n_1() || t_SMC0_1_s_0_n_2())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | y[y[i]] > 0.
            private boolean t_SMC0_2_s_0_n_0() {
                lock_ids[0] = target_locks[5] = 4 + i; // Acquire y[i]
                lock_ids[1] = target_locks[6] = 4 + 0; // Acquire y[0]
                lock_ids[2] = target_locks[7] = 4 + 1; // Acquire y[1]
                lock_ids[3] = target_locks[8] = 4 + 2; // Acquire y[2]
                lockManager.acquire_locks(lock_ids, 4);
                if(y[y[i]] > 0) {
                    lock_ids[0] = target_locks[0]; // Release i
                    lock_ids[1] = target_locks[5]; // Release y[i]
                    lock_ids[2] = target_locks[6]; // Release y[0]
                    lock_ids[3] = target_locks[7]; // Release y[1]
                    lock_ids[4] = target_locks[8]; // Release y[2]
                    lockManager.release_locks(lock_ids, 5);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[5]; // Release y[i]
                lock_ids[2] = target_locks[6]; // Release y[0]
                lock_ids[3] = target_locks[7]; // Release y[1]
                lock_ids[4] = target_locks[8]; // Release y[2]
                lockManager.release_locks(lock_ids, 5);
                return false;
            }

            // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | y[y[i]] > 0.
            private boolean execute_transition_SMC0_2() {
                // SLCO expression | y[y[i]] > 0.
                if(!(t_SMC0_2_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // Attempt to fire a transition starting in state SMC0.
            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 3 and x[i].
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | x[0] or x[1] or x[2].
                if(execute_transition_SMC0_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | y[y[i]] > 0.
                if(execute_transition_SMC0_2()) {
                    return;
                }
                // [SEQ.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 60) {
                    switch(currentState) {
                        case SMC0 -> exec_SMC0();
                    }
                }
            }

            // The thread's run method.
            public void run() {
                try {
                    exec();
                } catch(Exception e) {
                    lockManager.exception_unlock();
                    throw e;
                }
            }
        }

        // Start all threads.
        public void startThreads() {
            T_SM1.start();
        }

        // Join all threads.
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

    // Start all threads.
    private void startThreads() {
        for(SLCO_Class object : objects) {
            object.startThreads();
        }
    }

    // Join all threads.
    private void joinThreads() {
        for(SLCO_Class object : objects) {
            object.joinThreads();
        }
    }

    // Run the application.
    public static void main(String[] args) {
        // Initialize the model and execute.
        LockExample model = new LockExample();
        model.startThreads();
        model.joinThreads();
    }
}