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

        // Lock check
        void check_no_locks() {
            for(ReentrantLock lock: locks) {
                if(lock.isHeldByCurrentThread()) {
                    throw new RuntimeException("The locking structure is incorrect. locks remain at the start of an iteration.");
                }
            }
        }

        // Lock check
        void check_lock(int lock_id) {
            if(!locks[lock_id].isHeldByCurrentThread()) {
                throw new RuntimeException("Atomicity is violated due to not having locked a class variable.");
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

        // Global variables
        private volatile boolean[] x;
        private volatile int y;

        // Define the states fot the state machine SM1
        interface P_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1, 
                NewState0
            }
        }

        // Representation of the SLCO state machine SM1
        class SM1Thread extends Thread implements P_SM1Thread_States {
            // Current state
            private SM1Thread.States currentState;

            // Random number generator to handle non-determinism
            private final Random random;

            // Thread local variables
            private int i;
            private int j;
            private int k;

            // The lock manager
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused
            private final int[] lock_ids;
            private final int[] target_locks;

            SM1Thread (LockManager lockManagerInstance) {
                currentState = SM1Thread.States.SMC0;
                lockManager = lockManagerInstance;
                lock_ids = new int[1];
                target_locks = new int[1];
                random = new Random();
            }

            // SLCO expression wrapper | x[0] > 0
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 1 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(1 + 0); // Check x[0]
                if(x[0] > 0) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lock_ids[1] = target_locks[0]; // Release x[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                return false;
            }

            // SLCO transition (id:0, p:0) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | x[0] > 0
                if(!(t_SMC0_0_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | x[0] > 0
            private boolean t_SMC0_1_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 1 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(1 + 0); // Check x[0]
                if(x[0] > 0) {
                    lock_ids[0] = target_locks[0]; // Release x[0]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:1, p:1) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_1() {
                // SLCO expression | x[0] > 0
                if(!(t_SMC0_1_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // (Superfluous) SLCO transition (id:2, p:0) | SMC0 -> SMC0 | Excluded due to False guard

            // SLCO expression wrapper | x[0] <= 0
            private boolean t_SMC0_3_s_0_n_0() {
                lockManager.check_lock(1 + 0); // Check x[0]
                if(x[0] <= 0) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lock_ids[1] = target_locks[0]; // Release x[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:3, p:0) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_3() {
                // SLCO expression | x[0] <= 0
                if(!(t_SMC0_3_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | y <= 0
            private boolean t_SMC0_4_s_0_n_0() {
                lockManager.check_lock(0); // Check y
                if(y <= 0) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (id:4, p:0) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_4() {
                // SLCO expression | y <= 0
                if(!(t_SMC0_4_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | y > 0
            private boolean t_SMC0_5_s_0_n_0() {
                lockManager.check_lock(0); // Check y
                return y > 0;
            }

            // SLCO expression wrapper | y <= 3
            private boolean t_SMC0_5_s_0_n_1() {
                lockManager.check_lock(0); // Check y
                if(y <= 3) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (id:5, p:0) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_5() {
                // SLCO expression | y > 0 and y <= 3
                if(!(t_SMC0_5_s_0_n_0() && t_SMC0_5_s_0_n_1())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | y > 3
            private boolean t_SMC0_6_s_0_n_0() {
                lockManager.check_lock(0); // Check y
                if(y > 3) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:6, p:0) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_6() {
                // SLCO expression | y > 3
                if(!(t_SMC0_6_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (id:7, p:0) | SMC0 -> NewState0
            private boolean execute_transition_SMC0_7() {
                // (Superfluous) SLCO expression | true

                // SLCO assignment | [y := 0] -> y := 0
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check y
                y = 0;
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.NewState0;
                return true;
            }

            // SLCO transition (id:8, p:1) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_8() {
                // (Superfluous) SLCO expression | true

                // SLCO assignment | [y := 1] -> y := 1
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check y
                y = 1;
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (id:9, p:5) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_9() {
                // (Superfluous) SLCO expression | true

                // SLCO assignment | [y := 5] -> y := 5
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check y
                y = 5;
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (id:10, p:3) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_10() {
                // (Superfluous) SLCO expression | true

                // SLCO assignment | [y := 3] -> y := 3
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check y
                y = 3;
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (id:11, p:2) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_11() {
                // (Superfluous) SLCO expression | true

                // SLCO assignment | [y := 2] -> y := 2
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check y
                y = 2;
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (id:12, p:4) | SMC0 -> SMC0
            private boolean execute_transition_SMC0_12() {
                // (Superfluous) SLCO expression | true

                // SLCO assignment | [y := 4] -> y := 4
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check y
                y = 4;
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (id:0, p:0) | SMC1 -> SMC1
            private boolean execute_transition_SMC1_0() {
                // (Superfluous) SLCO expression | y > 3 or y < 5 -> true

                currentState = SM1Thread.States.SMC1;
                return true;
            }

            // (Superfluous) SLCO transition (id:1, p:0) | SMC1 -> SMC1 | Excluded due to False guard

            // SLCO expression wrapper | y > 3
            private boolean t_NewState0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check y
                if(y > 3) {
                    lock_ids[0] = target_locks[0]; // Release y
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState0 -> SMC0
            private boolean execute_transition_NewState0_0() {
                // SLCO expression | y > 3
                if(!(t_NewState0_0_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            private void exec_SMC0() {
                // [SEQ.START]
                // [DET.START]
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SMC0 -> SMC0 | guard: x[0] > 0
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // Excluded transitions:
                // - (Superfluous) SLCO transition (id:2, p:0) | SMC0 -> SMC0
                // [SEQ.END]
                // SLCO transition (id:3, p:0) | SMC0 -> SMC0 | guard: x[0] <= 0
                if(execute_transition_SMC0_3()) {
                    return;
                }
                // [DET.END]
                // [DET.START]
                // SLCO transition (id:4, p:0) | SMC0 -> SMC0 | guard: y <= 0
                if(execute_transition_SMC0_4()) {
                    return;
                }
                // SLCO transition (id:5, p:0) | SMC0 -> SMC0 | guard: y > 0 and y <= 3
                if(execute_transition_SMC0_5()) {
                    return;
                }
                // SLCO transition (id:6, p:0) | SMC0 -> SMC0 | guard: y > 3
                if(execute_transition_SMC0_6()) {
                    return;
                }
                // [DET.END]
                // SLCO transition (id:7, p:0) | SMC0 -> NewState0 | guard: true
                if(execute_transition_SMC0_7()) {
                    return;
                }
                // SLCO transition (id:1, p:1) | SMC0 -> SMC0 | guard: x[0] > 0
                if(execute_transition_SMC0_1()) {
                    return;
                }
                // SLCO transition (id:8, p:1) | SMC0 -> SMC0 | guard: true
                if(execute_transition_SMC0_8()) {
                    return;
                }
                // SLCO transition (id:11, p:2) | SMC0 -> SMC0 | guard: true
                if(execute_transition_SMC0_11()) {
                    return;
                }
                // SLCO transition (id:10, p:3) | SMC0 -> SMC0 | guard: true
                if(execute_transition_SMC0_10()) {
                    return;
                }
                // SLCO transition (id:12, p:4) | SMC0 -> SMC0 | guard: true
                if(execute_transition_SMC0_12()) {
                    return;
                }
                // SLCO transition (id:9, p:5) | SMC0 -> SMC0 | guard: true
                if(execute_transition_SMC0_9()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_SMC1() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SMC1 -> SMC1 | guard: true
                if(execute_transition_SMC1_0()) {
                    return;
                }
                // Excluded transitions:
                // - (Superfluous) SLCO transition (id:1, p:0) | SMC1 -> SMC1
                // [SEQ.END]
            }

            private void exec_NewState0() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | NewState0 -> SMC0 | guard: y > 3
                if(execute_transition_NewState0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Execute method
            private void exec() {
                while(true) {
                    lockManager.check_no_locks();
                    switch(currentState) {
                        case SMC0 -> exec_SMC0();
                        case SMC1 -> exec_SMC1();
                        case NewState0 -> exec_NewState0();
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

        P(boolean[] x, int y) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(3);

            // Instantiate global variables
            this.x = x;
            this.y = y;

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
                new boolean[]{ false, true },
                1
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