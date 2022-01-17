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
        private volatile int[] x;
        private volatile int[] y;
        private volatile boolean[] b;
        private volatile int i;

        // Define the states fot the state machine SM1
        interface P_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1, 
                NewState0, 
                NewState1, 
                NewState2, 
                NewState3
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
                lock_ids = new int[1];
                target_locks = new int[1];
                random = new Random();
            }

            // SLCO expression wrapper | x[0] > 0
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                if(x[0] > 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | SMC0 -> NewState3
            private boolean execute_transition_SMC0_0() {
                // SLCO composite | [x[0] > 0; i := i + 1; x[i] := 1]
                // SLCO expression | x[0] > 0
                if(!(t_SMC0_0_s_0_n_0())) {
                    return false;
                }
                // SLCO assignment | i := i + 1
                lockManager.check_lock(0); // Check statement lock
                i = i + 1;
                // SLCO assignment | x[i] := 1
                lockManager.check_lock(0); // Check statement lock
                x[i] = 1;
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.NewState3;
                return true;
            }

            // SLCO expression wrapper | x[0] > 0
            private boolean t_NewState0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                if(x[0] > 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState0 -> SMC0
            private boolean execute_transition_NewState0_0() {
                // SLCO composite | [x[0] > 0; i := 2; x[i] := 1; x[1] := 1]
                // SLCO expression | x[0] > 0
                if(!(t_NewState0_0_s_0_n_0())) {
                    return false;
                }
                // SLCO assignment | i := 2
                lockManager.check_lock(0); // Check statement lock
                i = 2;
                // SLCO assignment | x[i] := 1
                lockManager.check_lock(0); // Check statement lock
                x[i] = 1;
                // SLCO assignment | x[1] := 1
                lockManager.check_lock(0); // Check statement lock
                x[1] = 1;
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | x[0] > 0
            private boolean t_NewState1_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                if(x[0] > 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState1 -> NewState0
            private boolean execute_transition_NewState1_0() {
                // SLCO composite | [x[0] > 0; i := 0; x[i] := 1; x[1] := 1]
                // SLCO expression | x[0] > 0
                if(!(t_NewState1_0_s_0_n_0())) {
                    return false;
                }
                // SLCO assignment | i := 0
                lockManager.check_lock(0); // Check statement lock
                i = 0;
                // SLCO assignment | x[i] := 1
                lockManager.check_lock(0); // Check statement lock
                x[i] = 1;
                // SLCO assignment | x[1] := 1
                lockManager.check_lock(0); // Check statement lock
                x[1] = 1;
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.NewState0;
                return true;
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_NewState2_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 3
            private boolean t_NewState2_0_s_0_n_1() {
                lockManager.check_lock(0); // Check statement lock
                if(i < 3) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | b[i]
            private boolean t_NewState2_0_s_0_n_2() {
                lockManager.check_lock(0); // Check statement lock
                if(b[i]) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState2 -> NewState1
            private boolean execute_transition_NewState2_0() {
                // SLCO composite | [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1]
                // SLCO expression | i >= 0 and i < 3 and b[i]
                if(!(t_NewState2_0_s_0_n_0() && t_NewState2_0_s_0_n_1() && t_NewState2_0_s_0_n_2())) {
                    return false;
                }
                // SLCO assignment | i := 0
                lockManager.check_lock(0); // Check statement lock
                i = 0;
                // SLCO assignment | x[i] := 1
                lockManager.check_lock(0); // Check statement lock
                x[i] = 1;
                // SLCO assignment | x[1] := 1
                lockManager.check_lock(0); // Check statement lock
                x[1] = 1;
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.NewState1;
                return true;
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_NewState3_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_NewState3_0_s_0_n_1() {
                lockManager.check_lock(0); // Check statement lock
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState3 -> NewState2
            private boolean execute_transition_NewState3_0() {
                // SLCO composite | [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0]
                // SLCO expression | i >= 0 and i < 2
                if(!(t_NewState3_0_s_0_n_0() && t_NewState3_0_s_0_n_1())) {
                    return false;
                }
                // SLCO assignment | i := 0
                lockManager.check_lock(0); // Check statement lock
                i = 0;
                // SLCO assignment | x[i] := 0
                lockManager.check_lock(0); // Check statement lock
                x[i] = 0;
                // SLCO assignment | x[x[i]] := 0
                lockManager.check_lock(0); // Check statement lock
                x[x[i]] = 0;
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.NewState2;
                return true;
            }

            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SMC0 -> NewState3 | guard: [x[0] > 0; i := i + 1; x[i] := 1]
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            private void exec_NewState0() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | NewState0 -> SMC0 | guard: [x[0] > 0; i := 2; x[i] := 1; x[1] := 1]
                if(execute_transition_NewState0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_NewState1() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | NewState1 -> NewState0 | guard: [x[0] > 0; i := 0; x[i] := 1; x[1] := 1]
                if(execute_transition_NewState1_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_NewState2() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | NewState2 -> NewState1 | guard: [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1]
                if(execute_transition_NewState2_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_NewState3() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | NewState3 -> NewState2 | guard: [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0]
                if(execute_transition_NewState3_0()) {
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
                        case NewState1 -> exec_NewState1();
                        case NewState2 -> exec_NewState2();
                        case NewState3 -> exec_NewState3();
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

        P(int[] x, int[] y, boolean[] b, int i) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(6);

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
                new int[]{ 0, 0 },
                new int[]{ 0, 0 },
                new boolean[]{ false, false, false },
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