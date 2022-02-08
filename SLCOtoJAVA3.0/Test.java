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

    // Template for SLCO classes
    interface SLCO_Class {
        void startThreads();
        void joinThreads();
    }

    // Representation of the SLCO class P
    private static class P implements SLCO_Class {
        // The threads
        private final Thread T_SM1;
        private final Thread T_SM2;

        // Global variables
        private volatile int[] x;
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

            // A list of lock ids and target locks that can be reused
            private final int[] lock_ids;
            private final int[] target_locks;

            SM1Thread (LockManager lockManagerInstance) {
                currentState = SM1Thread.States.SMC0;
                lockManager = lockManagerInstance;
                lock_ids = new int[2];
                target_locks = new int[2];
                random = new Random();
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SMC0_0_s_0_n_1() {
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | x[i] = 0
            private boolean t_SMC0_0_s_0_n_2() {
                lock_ids[0] = target_locks[1] = 1 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                if(x[i] == 0) {
                    lock_ids[0] = target_locks[1]; // Release x[i]
                    lock_ids[1] = target_locks[0]; // Release i
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release x[i]
                lock_ids[1] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | i >= 0 and i < 2 and x[i] = 0
                if(!(t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1() && t_SMC0_0_s_0_n_2())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SMC0 -> SMC0 | guard: i >= 0 and i < 2 and x[i] = 0
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [SEQ.END]
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
                try {
                    exec();
                } catch(Exception e) {
                    lockManager.exception_unlock();
                    throw e;
                }
            }
        }

        // Define the states fot the state machine SM2
        interface P_SM2Thread_States {
            enum States {
                SMC0, 
                SMC1
            }
        }

        // Representation of the SLCO state machine SM2
        class SM2Thread extends Thread implements P_SM2Thread_States {
            // Current state
            private SM2Thread.States currentState;

            // Random number generator to handle non-determinism
            private final Random random;

            // Thread local variables
            private int[] y;

            // The lock manager
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused
            private final int[] lock_ids;
            private final int[] target_locks;

            SM2Thread (LockManager lockManagerInstance) {
                currentState = SM2Thread.States.SMC0;
                lockManager = lockManagerInstance;
                lock_ids = new int[1];
                target_locks = new int[2];
                random = new Random();
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SMC0_0_s_0_n_1() {
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2; x[i] := 0; y[i] := 0]
            private boolean execute_transition_SMC0_0() {
                // SLCO composite | [i >= 0 and i < 2; x[i] := 0; y[i] := 0]
                // SLCO expression | i >= 0 and i < 2
                if(!(t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1())) {
                    return false;
                }
                // SLCO assignment | x[i] := 0
                lock_ids[0] = target_locks[1] = 1 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                x[i] = 0;
                lock_ids[0] = target_locks[1]; // Release x[i]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y[i] := 0
                y[i] = 0;
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);

                currentState = SM2Thread.States.SMC0;
                return true;
            }

            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SMC0 -> SMC0 | guard: [i >= 0 and i < 2; x[i] := 0; y[i] := 0]
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [SEQ.END]
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
                try {
                    exec();
                } catch(Exception e) {
                    lockManager.exception_unlock();
                    throw e;
                }
            }
        }

        P(int[] x, int i) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(3);

            // Instantiate global variables
            this.x = x;
            this.i = i;

            // Instantiate state machines
            T_SM1 = new SM1Thread(lockManager);
            T_SM2 = new SM2Thread(lockManager);
        }

        // Start all threads
        public void startThreads() {
            T_SM1.start();
            T_SM2.start();
        }

        // Join all threads
        public void joinThreads() {
            while (true) {
                try {
                    T_SM1.join();
                    T_SM2.join();
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
                new int[]{ 0, 0 },
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