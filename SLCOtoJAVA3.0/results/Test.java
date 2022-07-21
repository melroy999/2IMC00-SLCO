package testing;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.time.Duration;
import java.time.Instant;

// SLCO model Test.
public class Test {
    // The objects in the model.
    private final SLCO_Class[] objects;

    // Interface for SLCO classes.
    interface SLCO_Class {
        void startThreads();
        void joinThreads();
    }

    Test() {
        // Instantiate the objects.
        objects = new SLCO_Class[] {
            new P(
                new int[]{ 0, 0 },
                new int[]{ 0, 0 },
                new boolean[]{ false, false, false },
                0,
                new int[]{ 0, 0 },
                (char) 0
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
        private final int[] x;
        private final int[] y;
        private final boolean[] b;
        private volatile int i;
        private final int[] c;
        private volatile int d;

        P(int[] x, int[] y, boolean[] b, int i, int[] c, int d) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(11);

            // Instantiate the class variables.
            this.x = x;
            this.y = y;
            this.b = b;
            this.i = i;
            this.c = c;
            this.d = d;

            // Instantiate the state machine threads and pass on the class' lock manager.
            T_SM1 = new P_SM1Thread(lockManager);
        }

        // Define the states fot the state machine SM1.
        interface P_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1, 
                SMC2, 
                SMC3
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
                lock_ids = new int[4];
                target_locks = new int[4];
                random = new Random();
            }

            // SLCO expression wrapper | x[i] > 0.
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 8; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[1] = 9 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                if(x[i] > 0) {
                    lock_ids[0] = target_locks[1]; // Release x[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[1]; // Release x[i]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | x[i + 1] > 0.
            private boolean t_SMC0_0_s_0_n_1() {
                lock_ids[0] = target_locks[2] = 9 + i + 1; // Acquire x[i + 1]
                lockManager.acquire_locks(lock_ids, 1);
                if(x[i + 1] > 0) {
                    lock_ids[0] = target_locks[0]; // Release i
                    lock_ids[1] = target_locks[2]; // Release x[i + 1]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[2]; // Release x[i + 1]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | x[i] > 0 and x[i + 1] > 0.
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | x[i] > 0 and x[i + 1] > 0.
                if(!(t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | x[i] > 0.
            private boolean t_SMC1_0_s_0_n_0() {
                lock_ids[0] = target_locks[3] = 8; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 9 + i + 1; // Acquire x[i + 1]
                lock_ids[1] = target_locks[1] = 9 + 0; // Acquire x[0]
                lock_ids[2] = target_locks[2] = 9 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 3);
                if(x[i] > 0) {
                    lock_ids[0] = target_locks[2]; // Release x[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x[i + 1]
                lock_ids[1] = target_locks[1]; // Release x[0]
                lock_ids[2] = target_locks[2]; // Release x[i]
                lock_ids[3] = target_locks[3]; // Release i
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO expression wrapper | x[i + 1] > 0.
            private boolean t_SMC1_0_s_0_n_1() {
                if(x[i + 1] > 0) {
                    lock_ids[0] = target_locks[0]; // Release x[i + 1]
                    lock_ids[1] = target_locks[3]; // Release i
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x[i + 1]
                lock_ids[1] = target_locks[1]; // Release x[0]
                lock_ids[2] = target_locks[3]; // Release i
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | x[0] > 0.
            private boolean t_SMC1_0_s_0_n_2() {
                if(x[0] > 0) {
                    lock_ids[0] = target_locks[1]; // Release x[0]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release x[0]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC1 -> SMC1 | x[i] > 0 and x[i + 1] > 0 and x[0] > 0.
            private boolean execute_transition_SMC1_0() {
                // SLCO expression | x[i] > 0 and x[i + 1] > 0 and x[0] > 0.
                if(!(t_SMC1_0_s_0_n_0() && t_SMC1_0_s_0_n_1() && t_SMC1_0_s_0_n_2())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC1;
                return true;
            }

            // SLCO expression wrapper | x[0] > 0.
            private boolean t_SMC2_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 8; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[1] = 9 + i; // Acquire x[i]
                lock_ids[1] = target_locks[2] = 9 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 2);
                if(x[0] > 0) {
                    lock_ids[0] = target_locks[2]; // Release x[0]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[1]; // Release x[i]
                lock_ids[2] = target_locks[2]; // Release x[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | x[i] > 0.
            private boolean t_SMC2_0_s_0_n_1() {
                if(x[i] > 0) {
                    lock_ids[0] = target_locks[0]; // Release i
                    lock_ids[1] = target_locks[1]; // Release x[i]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[1]; // Release x[i]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC2 -> SMC2 | x[0] > 0 and x[i] > 0.
            private boolean execute_transition_SMC2_0() {
                // SLCO expression | x[0] > 0 and x[i] > 0.
                if(!(t_SMC2_0_s_0_n_0() && t_SMC2_0_s_0_n_1())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC2;
                return true;
            }

            // SLCO expression wrapper | x[0] > 0.
            private boolean t_SMC3_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 8; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[1] = 9 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 1);
                if(x[0] > 0) {
                    lock_ids[0] = target_locks[1]; // Release x[0]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[1]; // Release x[0]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | x[1 + i * i] > 0.
            private boolean t_SMC3_0_s_0_n_1() {
                lock_ids[0] = target_locks[2] = 9 + 1 + i * i; // Acquire x[1 + i * i]
                lockManager.acquire_locks(lock_ids, 1);
                if(x[1 + i * i] > 0) {
                    lock_ids[0] = target_locks[0]; // Release i
                    lock_ids[1] = target_locks[2]; // Release x[1 + i * i]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[2]; // Release x[1 + i * i]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC3 -> SMC3 | x[0] > 0 and x[1 + i * i] > 0.
            private boolean execute_transition_SMC3_0() {
                // SLCO expression | x[0] > 0 and x[1 + i * i] > 0.
                if(!(t_SMC3_0_s_0_n_0() && t_SMC3_0_s_0_n_1())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC3;
                return true;
            }

            // Attempt to fire a transition starting in state SMC0.
            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | x[i] > 0 and x[i + 1] > 0.
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SMC1.
            private void exec_SMC1() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC1 -> SMC1 | x[i] > 0 and x[i + 1] > 0 and x[0] > 0.
                if(execute_transition_SMC1_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SMC2.
            private void exec_SMC2() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC2 -> SMC2 | x[0] > 0 and x[i] > 0.
                if(execute_transition_SMC2_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SMC3.
            private void exec_SMC3() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC3 -> SMC3 | x[0] > 0 and x[1 + i * i] > 0.
                if(execute_transition_SMC3_0()) {
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
                        case SMC1 -> exec_SMC1();
                        case SMC2 -> exec_SMC2();
                        case SMC3 -> exec_SMC3();
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
        Test model = new Test();
        model.startThreads();
        model.joinThreads();
    }
}