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
        private volatile char[] c;
        private volatile char d;

        // Define the states fot the state machine SM1
        interface P_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1, 
                SimpleState0, 
                SimpleState1, 
                SimpleState2, 
                SimpleState3, 
                SimpleState4, 
                SimpleState5, 
                SimpleState6, 
                SimpleState7
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

            // SLCO transition (p:0, id:0) | SMC0 -> SimpleState0 | [x[0] > 0; i := i + 1; x[i] := 1]
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

                currentState = SM1Thread.States.SimpleState0;
                return true;
            }

            // SLCO expression wrapper | x[0] > 0
            private boolean t_SimpleState0_0_s_0_n_0() {
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

            // SLCO transition (p:0, id:0) | SimpleState0 -> SimpleState1 | [x[0] > 0; i := 2; x[i] := 1; x[1] := 1]
            private boolean execute_transition_SimpleState0_0() {
                // SLCO composite | [x[0] > 0; i := 2; x[i] := 1; x[1] := 1]
                // SLCO expression | x[0] > 0
                if(!(t_SimpleState0_0_s_0_n_0())) {
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

                currentState = SM1Thread.States.SimpleState1;
                return true;
            }

            // SLCO expression wrapper | x[0] > 0
            private boolean t_SimpleState1_0_s_0_n_0() {
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

            // SLCO transition (p:0, id:0) | SimpleState1 -> SimpleState2 | [x[0] > 0; i := 0; x[i] := 1; x[1] := 1]
            private boolean execute_transition_SimpleState1_0() {
                // SLCO composite | [x[0] > 0; i := 0; x[i] := 1; x[1] := 1]
                // SLCO expression | x[0] > 0
                if(!(t_SimpleState1_0_s_0_n_0())) {
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

                currentState = SM1Thread.States.SimpleState2;
                return true;
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SimpleState2_0_s_0_n_0() {
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
            private boolean t_SimpleState2_0_s_0_n_1() {
                lockManager.check_lock(0); // Check statement lock
                if(i < 3) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | b[i]
            private boolean t_SimpleState2_0_s_0_n_2() {
                lockManager.check_lock(0); // Check statement lock
                if(b[i]) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:0) | SimpleState2 -> SimpleState3 | [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1]
            private boolean execute_transition_SimpleState2_0() {
                // SLCO composite | [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1]
                // SLCO expression | i >= 0 and i < 3 and b[i]
                if(!(t_SimpleState2_0_s_0_n_0() && t_SimpleState2_0_s_0_n_1() && t_SimpleState2_0_s_0_n_2())) {
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

                currentState = SM1Thread.States.SimpleState3;
                return true;
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SimpleState3_0_s_0_n_0() {
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
            private boolean t_SimpleState3_0_s_0_n_1() {
                lockManager.check_lock(0); // Check statement lock
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:0) | SimpleState3 -> SimpleState4 | [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0]
            private boolean execute_transition_SimpleState3_0() {
                // SLCO composite | [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0]
                // SLCO expression | i >= 0 and i < 2
                if(!(t_SimpleState3_0_s_0_n_0() && t_SimpleState3_0_s_0_n_1())) {
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

                currentState = SM1Thread.States.SimpleState4;
                return true;
            }

            // SLCO expression wrapper | true
            private boolean t_SimpleState4_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                try {
                    return true;
                } finally {
                    lock_ids[0] = target_locks[0]; // Release statement lock
                    lockManager.release_locks(lock_ids, 1);
                }
            }

            // SLCO expression wrapper | true
            private boolean t_SimpleState4_0_s_1_n_1() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                try {
                    return true;
                } finally {
                    
                }
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SimpleState4_0_s_3_n_2() {
                lockManager.check_lock(0); // Check statement lock
                return i >= 0;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SimpleState4_0_s_3_n_3() {
                lockManager.check_lock(0); // Check statement lock
                return i < 2;
            }

            // SLCO expression wrapper | b[i]
            private boolean t_SimpleState4_0_s_3_n_4() {
                lockManager.check_lock(0); // Check statement lock
                return b[i];
            }

            // SLCO transition (p:0, id:0) | SimpleState4 -> SimpleState5 | true | [true; i := 0; b[i] := i >= 0 and i < 2 and b[i]]
            private boolean execute_transition_SimpleState4_0() {
                // SLCO expression | true
                if(!(t_SimpleState4_0_s_0_n_0())) {
                    return false;
                }

                // SLCO composite | [i := 0; b[i] := i >= 0 and i < 2 and b[i]] -> [true; i := 0; b[i] := i >= 0 and i < 2 and b[i]]
                // SLCO expression | true
                if(!(t_SimpleState4_0_s_1_n_1())) {
                    return false;
                }
                // SLCO assignment | i := 0
                lockManager.check_lock(0); // Check statement lock
                i = 0;
                // SLCO assignment | b[i] := i >= 0 and i < 2 and b[i]
                lockManager.check_lock(0); // Check statement lock
                b[i] = t_SimpleState4_0_s_3_n_2() && t_SimpleState4_0_s_3_n_3() && t_SimpleState4_0_s_3_n_4();
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.SimpleState5;
                return true;
            }

            // SLCO expression wrapper | true
            private boolean t_SimpleState5_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                try {
                    return true;
                } finally {
                    lock_ids[0] = target_locks[0]; // Release statement lock
                    lockManager.release_locks(lock_ids, 1);
                }
            }

            // SLCO expression wrapper | true
            private boolean t_SimpleState5_0_s_1_n_1() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                try {
                    return true;
                } finally {
                    
                }
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SimpleState5_0_s_4_n_2() {
                lockManager.check_lock(0); // Check statement lock
                return i >= 0;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SimpleState5_0_s_4_n_3() {
                lockManager.check_lock(0); // Check statement lock
                return i < 2;
            }

            // SLCO expression wrapper | b[i]
            private boolean t_SimpleState5_0_s_4_n_4() {
                lockManager.check_lock(0); // Check statement lock
                return b[i];
            }

            // SLCO transition (p:0, id:0) | SimpleState5 -> SimpleState6 | true | [true; b[0] := true; i := 0; b[i] := i >= 0 and i < 2 and b[i]]
            private boolean execute_transition_SimpleState5_0() {
                // SLCO expression | true
                if(!(t_SimpleState5_0_s_0_n_0())) {
                    return false;
                }

                // SLCO composite | [b[0] := true; i := 0; b[i] := i >= 0 and i < 2 and b[i]] -> [true; b[0] := true; i := 0; b[i] := i >= 0 and i < 2 and b[i]]
                // SLCO expression | true
                if(!(t_SimpleState5_0_s_1_n_1())) {
                    return false;
                }
                // SLCO assignment | b[0] := true
                lockManager.check_lock(0); // Check statement lock
                b[0] = true;
                // SLCO assignment | i := 0
                lockManager.check_lock(0); // Check statement lock
                i = 0;
                // SLCO assignment | b[i] := i >= 0 and i < 2 and b[i]
                lockManager.check_lock(0); // Check statement lock
                b[i] = t_SimpleState5_0_s_4_n_2() && t_SimpleState5_0_s_4_n_3() && t_SimpleState5_0_s_4_n_4();
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.SimpleState6;
                return true;
            }

            // SLCO expression wrapper | true
            private boolean t_SimpleState6_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                try {
                    return true;
                } finally {
                    lock_ids[0] = target_locks[0]; // Release statement lock
                    lockManager.release_locks(lock_ids, 1);
                }
            }

            // SLCO transition (p:0, id:0) | SimpleState6 -> SimpleState7 | true | d := 0
            private boolean execute_transition_SimpleState6_0() {
                // SLCO expression | true
                if(!(t_SimpleState6_0_s_0_n_0())) {
                    return false;
                }

                // SLCO assignment | d := 0
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                d = (0) & 0xff;
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.SimpleState7;
                return true;
            }

            // SLCO expression wrapper | true
            private boolean t_SimpleState7_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                try {
                    return true;
                } finally {
                    lock_ids[0] = target_locks[0]; // Release statement lock
                    lockManager.release_locks(lock_ids, 1);
                }
            }

            // SLCO transition (p:0, id:0) | SimpleState7 -> SMC0 | true | c[0] := 0
            private boolean execute_transition_SimpleState7_0() {
                // SLCO expression | true
                if(!(t_SimpleState7_0_s_0_n_0())) {
                    return false;
                }

                // SLCO assignment | c[0] := 0
                lock_ids[0] = target_locks[0] = 0; // Acquire statement lock
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check statement lock
                c[0] = (0) & 0xff;
                lock_ids[0] = target_locks[0]; // Release statement lock
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SMC0 -> SimpleState0 | guard: [x[0] > 0; i := i + 1; x[i] := 1]
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            private void exec_SimpleState0() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SimpleState0 -> SimpleState1 | guard: [x[0] > 0; i := 2; x[i] := 1; x[1] := 1]
                if(execute_transition_SimpleState0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_SimpleState1() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SimpleState1 -> SimpleState2 | guard: [x[0] > 0; i := 0; x[i] := 1; x[1] := 1]
                if(execute_transition_SimpleState1_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_SimpleState2() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SimpleState2 -> SimpleState3 | guard: [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1]
                if(execute_transition_SimpleState2_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_SimpleState3() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SimpleState3 -> SimpleState4 | guard: [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0]
                if(execute_transition_SimpleState3_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_SimpleState4() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SimpleState4 -> SimpleState5 | guard: true
                if(execute_transition_SimpleState4_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_SimpleState5() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SimpleState5 -> SimpleState6 | guard: true
                if(execute_transition_SimpleState5_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_SimpleState6() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SimpleState6 -> SimpleState7 | guard: true
                if(execute_transition_SimpleState6_0()) {
                    return;
                }
                // [SEQ.END]
            }

            private void exec_SimpleState7() {
                // [SEQ.START]
                // SLCO transition (id:0, p:0) | SimpleState7 -> SMC0 | guard: true
                if(execute_transition_SimpleState7_0()) {
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
                        case SimpleState0 -> exec_SimpleState0();
                        case SimpleState1 -> exec_SimpleState1();
                        case SimpleState2 -> exec_SimpleState2();
                        case SimpleState3 -> exec_SimpleState3();
                        case SimpleState4 -> exec_SimpleState4();
                        case SimpleState5 -> exec_SimpleState5();
                        case SimpleState6 -> exec_SimpleState6();
                        case SimpleState7 -> exec_SimpleState7();
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

        P(int[] x, int[] y, boolean[] b, int i, char[] c, char d) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(8);

            // Instantiate global variables
            this.x = x;
            this.y = y;
            this.b = b;
            this.i = i;
            this.c = c;
            this.d = d;

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
                0,
                new char[]{ 0, 0 },
                (char) 0
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