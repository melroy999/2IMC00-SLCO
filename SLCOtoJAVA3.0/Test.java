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
        private volatile int i;
        private volatile int j;

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
                target_locks = new int[4];
                random = new Random();
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[2] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check i
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SMC0_0_s_0_n_1() {
                lockManager.check_lock(0); // Check i
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | x[i] = 0
            private boolean t_SMC0_0_s_0_n_2() {
                lock_ids[0] = target_locks[3] = 4 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check i
                lockManager.check_lock(4 + i); // Check x[i]
                if(x[i] == 0) {
                    lock_ids[0] = target_locks[2]; // Release i
                    lock_ids[1] = target_locks[3]; // Release x[i]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release i
                lock_ids[1] = target_locks[3]; // Release x[i]
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

            // SLCO expression wrapper | y[j] = 0
            private boolean t_SMC0_1_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 1; // Acquire j
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 2 + j; // Acquire y[j]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(2 + j); // Check y[j]
                lockManager.check_lock(1); // Check j
                if(y[j] == 0) {
                    lock_ids[0] = target_locks[0]; // Release y[j]
                    lock_ids[1] = target_locks[1]; // Release j
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y[j]
                lock_ids[1] = target_locks[1]; // Release j
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | y[j] = 0
            private boolean execute_transition_SMC0_1() {
                // SLCO expression | y[j] = 0
                if(!(t_SMC0_1_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            private void exec_SMC0() {
                // [N_DET.START]
                switch(random.nextInt(2)) {
                    case 0 -> {
                        // SLCO transition (id:0, p:0) | SMC0 -> SMC0 | guard: i >= 0 and i < 2 and x[i] = 0
                        if(execute_transition_SMC0_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (id:1, p:0) | SMC0 -> SMC0 | guard: y[j] = 0
                        if(execute_transition_SMC0_1()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            // Execute method
            private void exec() {
                while(true) {
                    lockManager.check_no_locks();
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

        P(int[] x, int[] y, int i, int j) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(6);

            // Instantiate global variables
            this.x = x;
            this.y = y;
            this.i = i;
            this.j = j;

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

    // Representation of the SLCO class Q
    private static class Q implements SLCO_Class {
        // The threads
        private final Thread T_SM1;

        // Global variables
        private volatile int[] x;
        private volatile int[] y;
        private volatile int i;
        private volatile int j;

        // Define the states fot the state machine SM1
        interface Q_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1
            }
        }

        // Representation of the SLCO state machine SM1
        class SM1Thread extends Thread implements Q_SM1Thread_States {
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
                target_locks = new int[4];
                random = new Random();
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check i
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SMC0_0_s_0_n_1() {
                lockManager.check_lock(0); // Check i
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
                lockManager.check_lock(0); // Check i
                lockManager.check_lock(1 + i); // Check x[i]
                if(x[i] == 0) {
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

            // SLCO expression wrapper | y[j] = 0
            private boolean t_SMC0_0_s_0_n_3() {
                lock_ids[0] = target_locks[3] = 3; // Acquire j
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[2] = 4 + j; // Acquire y[j]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(4 + j); // Check y[j]
                lockManager.check_lock(3); // Check j
                if(y[j] == 0) {
                    lock_ids[0] = target_locks[2]; // Release y[j]
                    lock_ids[1] = target_locks[3]; // Release j
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release y[j]
                lock_ids[1] = target_locks[3]; // Release j
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0 and y[j] = 0
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | i >= 0 and i < 2 and x[i] = 0 and y[j] = 0
                if(!(t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1() && t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            private void exec_SMC0() {
                // [N_DET.START]
                // SLCO transition (id:0, p:0) | SMC0 -> SMC0 | guard: i >= 0 and i < 2 and x[i] = 0 and y[j] = 0
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [N_DET.END]
            }

            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            // Execute method
            private void exec() {
                while(true) {
                    lockManager.check_no_locks();
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

        Q(int[] x, int[] y, int i, int j) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(6);

            // Instantiate global variables
            this.x = x;
            this.y = y;
            this.i = i;
            this.j = j;

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

    // Representation of the SLCO class R
    private static class R implements SLCO_Class {
        // The threads
        private final Thread T_SM1;

        // Global variables
        private volatile int[] x;
        private volatile int[] y;
        private volatile int i;
        private volatile int j;

        // Define the states fot the state machine SM1
        interface R_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1
            }
        }

        // Representation of the SLCO state machine SM1
        class SM1Thread extends Thread implements R_SM1Thread_States {
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
                target_locks = new int[4];
                random = new Random();
            }

            // SLCO expression wrapper | y[j] = 0
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 0; // Acquire j
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 1 + j; // Acquire y[j]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(1 + j); // Check y[j]
                lockManager.check_lock(0); // Check j
                if(y[j] == 0) {
                    lock_ids[0] = target_locks[0]; // Release y[j]
                    lock_ids[1] = target_locks[1]; // Release j
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y[j]
                lock_ids[1] = target_locks[1]; // Release j
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SMC0_0_s_0_n_1() {
                lock_ids[0] = target_locks[2] = 3; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3); // Check i
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SMC0_0_s_0_n_2() {
                lockManager.check_lock(3); // Check i
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | x[i] = 0
            private boolean t_SMC0_0_s_0_n_3() {
                lock_ids[0] = target_locks[3] = 4 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3); // Check i
                lockManager.check_lock(4 + i); // Check x[i]
                if(x[i] == 0) {
                    lock_ids[0] = target_locks[2]; // Release i
                    lock_ids[1] = target_locks[3]; // Release x[i]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release i
                lock_ids[1] = target_locks[3]; // Release x[i]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | y[j] = 0 and i >= 0 and i < 2 and x[i] = 0
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | y[j] = 0 and i >= 0 and i < 2 and x[i] = 0
                if(!(t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1() && t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            private void exec_SMC0() {
                // [N_DET.START]
                // SLCO transition (id:0, p:0) | SMC0 -> SMC0 | guard: y[j] = 0 and i >= 0 and i < 2 and x[i] = 0
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [N_DET.END]
            }

            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            // Execute method
            private void exec() {
                while(true) {
                    lockManager.check_no_locks();
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

        R(int[] x, int[] y, int i, int j) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(6);

            // Instantiate global variables
            this.x = x;
            this.y = y;
            this.i = i;
            this.j = j;

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

    // Representation of the SLCO class S
    private static class S implements SLCO_Class {
        // The threads
        private final Thread T_SM1;

        // Global variables
        private volatile boolean[] x;
        private volatile boolean[] y;
        private volatile boolean[] z;
        private volatile boolean[] b;
        private volatile boolean[] c;
        private volatile int i;
        private volatile int j;

        // Define the states fot the state machine SM1
        interface S_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1
            }
        }

        // Representation of the SLCO state machine SM1
        class SM1Thread extends Thread implements S_SM1Thread_States {
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
                lock_ids = new int[6];
                target_locks = new int[13];
                random = new Random();
            }

            // SLCO expression wrapper | x[y[0]]
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 6 + 0; // Acquire x[0]
                lock_ids[1] = target_locks[5] = 6 + 1; // Acquire x[1]
                lockManager.acquire_locks(lock_ids, 2);
                lock_ids[0] = target_locks[3] = 8 + 0; // Acquire z[0]
                lock_ids[1] = target_locks[4] = 8 + x[0]; // Acquire z[x[0]]
                lockManager.acquire_locks(lock_ids, 2);
                lock_ids[0] = target_locks[2] = 10 + z[0]; // Acquire y[z[0]]
                lock_ids[1] = target_locks[1] = 10 + 0; // Acquire y[0]
                lockManager.acquire_locks(lock_ids, 2);
                lock_ids[0] = target_locks[6] = 6 + y[0]; // Acquire x[y[0]]
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[5]; // Release x[1]
                lockManager.release_locks(lock_ids, 1);
                lockManager.check_lock(10 + 0); // Check y[0]
                lockManager.check_lock(6 + y[0]); // Check x[y[0]]
                if(x[y[0]]) {
                    lock_ids[0] = target_locks[6]; // Release x[y[0]]
                    lock_ids[1] = target_locks[1]; // Release y[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[6]; // Release x[y[0]]
                lock_ids[1] = target_locks[0]; // Release x[0]
                lock_ids[2] = target_locks[2]; // Release y[z[0]]
                lock_ids[3] = target_locks[1]; // Release y[0]
                lock_ids[4] = target_locks[3]; // Release z[0]
                lock_ids[5] = target_locks[4]; // Release z[x[0]]
                lockManager.release_locks(lock_ids, 6);
                return false;
            }

            // SLCO expression wrapper | y[z[0]]
            private boolean t_SMC0_0_s_0_n_1() {
                lockManager.check_lock(10 + z[0]); // Check y[z[0]]
                lockManager.check_lock(8 + 0); // Check z[0]
                if(y[z[0]]) {
                    lock_ids[0] = target_locks[2]; // Release y[z[0]]
                    lock_ids[1] = target_locks[3]; // Release z[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[2]; // Release y[z[0]]
                lock_ids[2] = target_locks[3]; // Release z[0]
                lock_ids[3] = target_locks[4]; // Release z[x[0]]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO expression wrapper | z[x[0]]
            private boolean t_SMC0_0_s_0_n_2() {
                lockManager.check_lock(8 + x[0]); // Check z[x[0]]
                lockManager.check_lock(6 + 0); // Check x[0]
                if(z[x[0]]) {
                    lock_ids[0] = target_locks[4]; // Release z[x[0]]
                    lock_ids[1] = target_locks[0]; // Release x[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[4]; // Release z[x[0]]
                lock_ids[1] = target_locks[0]; // Release x[0]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | x[y[0]] and y[z[0]] and z[x[0]]
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | x[y[0]] and y[z[0]] and z[x[0]]
                if(!(t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1() && t_SMC0_0_s_0_n_2())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | b[j]
            private boolean t_SMC0_1_s_0_n_0() {
                lock_ids[0] = target_locks[9] = 0; // Acquire j
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[11] = 1 + 0; // Acquire b[0]
                lock_ids[1] = target_locks[10] = 1 + 1; // Acquire b[1]
                lock_ids[2] = target_locks[8] = 1 + j; // Acquire b[j]
                lockManager.acquire_locks(lock_ids, 3);
                lockManager.check_lock(1 + j); // Check b[j]
                lockManager.check_lock(0); // Check j
                if(b[j]) {
                    lock_ids[0] = target_locks[8]; // Release b[j]
                    lock_ids[1] = target_locks[9]; // Release j
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[11]; // Release b[0]
                lock_ids[1] = target_locks[10]; // Release b[1]
                lock_ids[2] = target_locks[8]; // Release b[j]
                lock_ids[3] = target_locks[9]; // Release j
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SMC0_1_s_0_n_1() {
                lock_ids[0] = target_locks[7] = 3; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3); // Check i
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lock_ids[1] = target_locks[10]; // Release b[1]
                lock_ids[2] = target_locks[11]; // Release b[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SMC0_1_s_0_n_2() {
                lockManager.check_lock(3); // Check i
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lock_ids[1] = target_locks[10]; // Release b[1]
                lock_ids[2] = target_locks[11]; // Release b[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | b[i]
            private boolean t_SMC0_1_s_0_n_3() {
                lockManager.check_lock(3); // Check i
                lockManager.check_lock(1 + i); // Check b[i]
                if(b[i]) {
                    lock_ids[0] = target_locks[7]; // Release i
                    lock_ids[1] = target_locks[10]; // Release b[1]
                    lock_ids[2] = target_locks[11]; // Release b[0]
                    lockManager.release_locks(lock_ids, 3);
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lock_ids[1] = target_locks[10]; // Release b[1]
                lock_ids[2] = target_locks[11]; // Release b[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | b[j] and i >= 0 and i < 2 and b[i]
            private boolean execute_transition_SMC0_1() {
                // SLCO expression | b[j] and i >= 0 and i < 2 and b[i]
                if(!(t_SMC0_1_s_0_n_0() && t_SMC0_1_s_0_n_1() && t_SMC0_1_s_0_n_2() && t_SMC0_1_s_0_n_3())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SMC0_2_s_0_n_0() {
                lock_ids[0] = target_locks[7] = 3; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3); // Check i
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SMC0_2_s_0_n_1() {
                lockManager.check_lock(3); // Check i
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | c[i]
            private boolean t_SMC0_2_s_0_n_2() {
                lock_ids[0] = target_locks[12] = 4 + i; // Acquire c[i]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3); // Check i
                lockManager.check_lock(4 + i); // Check c[i]
                if(c[i]) {
                    lock_ids[0] = target_locks[7]; // Release i
                    lock_ids[1] = target_locks[12]; // Release c[i]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lock_ids[1] = target_locks[12]; // Release c[i]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | i >= 0 and i < 2 and c[i]
            private boolean execute_transition_SMC0_2() {
                // SLCO expression | i >= 0 and i < 2 and c[i]
                if(!(t_SMC0_2_s_0_n_0() && t_SMC0_2_s_0_n_1() && t_SMC0_2_s_0_n_2())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            private void exec_SMC0() {
                // [N_DET.START]
                switch(random.nextInt(3)) {
                    case 0 -> {
                        // SLCO transition (id:0, p:0) | SMC0 -> SMC0 | guard: x[y[0]] and y[z[0]] and z[x[0]]
                        if(execute_transition_SMC0_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (id:1, p:0) | SMC0 -> SMC0 | guard: b[j] and i >= 0 and i < 2 and b[i]
                        if(execute_transition_SMC0_1()) {
                            return;
                        }
                    }
                    case 2 -> {
                        // SLCO transition (id:2, p:0) | SMC0 -> SMC0 | guard: i >= 0 and i < 2 and c[i]
                        if(execute_transition_SMC0_2()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            // Execute method
            private void exec() {
                while(true) {
                    lockManager.check_no_locks();
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

        S(boolean[] x, boolean[] y, boolean[] z, boolean[] b, boolean[] c, int i, int j) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(12);

            // Instantiate global variables
            this.x = x;
            this.y = y;
            this.z = z;
            this.b = b;
            this.c = c;
            this.i = i;
            this.j = j;

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

    // Representation of the SLCO class T
    private static class T implements SLCO_Class {
        // The threads
        private final Thread T_SM1;

        // Global variables
        private volatile boolean[] x;
        private volatile boolean[] y;
        private volatile boolean[] z;
        private volatile boolean[] b;
        private volatile boolean[] c;
        private volatile int i;
        private volatile int j;

        // Define the states fot the state machine SM1
        interface T_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1
            }
        }

        // Representation of the SLCO state machine SM1
        class SM1Thread extends Thread implements T_SM1Thread_States {
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
                lock_ids = new int[6];
                target_locks = new int[13];
                random = new Random();
            }

            // SLCO expression wrapper | x[y[0]]
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 6 + 0; // Acquire x[0]
                lock_ids[1] = target_locks[5] = 6 + 1; // Acquire x[1]
                lockManager.acquire_locks(lock_ids, 2);
                lock_ids[0] = target_locks[3] = 8 + 0; // Acquire z[0]
                lock_ids[1] = target_locks[4] = 8 + x[0]; // Acquire z[x[0]]
                lockManager.acquire_locks(lock_ids, 2);
                lock_ids[0] = target_locks[2] = 10 + z[0]; // Acquire y[z[0]]
                lock_ids[1] = target_locks[1] = 10 + 0; // Acquire y[0]
                lockManager.acquire_locks(lock_ids, 2);
                lock_ids[0] = target_locks[6] = 6 + y[0]; // Acquire x[y[0]]
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[5]; // Release x[1]
                lockManager.release_locks(lock_ids, 1);
                lockManager.check_lock(10 + 0); // Check y[0]
                lockManager.check_lock(6 + y[0]); // Check x[y[0]]
                if(x[y[0]]) {
                    lock_ids[0] = target_locks[6]; // Release x[y[0]]
                    lock_ids[1] = target_locks[1]; // Release y[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[6]; // Release x[y[0]]
                lock_ids[1] = target_locks[0]; // Release x[0]
                lock_ids[2] = target_locks[2]; // Release y[z[0]]
                lock_ids[3] = target_locks[1]; // Release y[0]
                lock_ids[4] = target_locks[3]; // Release z[0]
                lock_ids[5] = target_locks[4]; // Release z[x[0]]
                lockManager.release_locks(lock_ids, 6);
                return false;
            }

            // SLCO expression wrapper | y[z[0]]
            private boolean t_SMC0_0_s_0_n_1() {
                lockManager.check_lock(10 + z[0]); // Check y[z[0]]
                lockManager.check_lock(8 + 0); // Check z[0]
                if(y[z[0]]) {
                    lock_ids[0] = target_locks[2]; // Release y[z[0]]
                    lock_ids[1] = target_locks[3]; // Release z[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[2]; // Release y[z[0]]
                lock_ids[2] = target_locks[3]; // Release z[0]
                lock_ids[3] = target_locks[4]; // Release z[x[0]]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO expression wrapper | z[x[0]]
            private boolean t_SMC0_0_s_0_n_2() {
                lockManager.check_lock(8 + x[0]); // Check z[x[0]]
                lockManager.check_lock(6 + 0); // Check x[0]
                if(z[x[0]]) {
                    lock_ids[0] = target_locks[4]; // Release z[x[0]]
                    lock_ids[1] = target_locks[0]; // Release x[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[4]; // Release z[x[0]]
                lock_ids[1] = target_locks[0]; // Release x[0]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | x[y[0]] and y[z[0]] and z[x[0]]
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | x[y[0]] and y[z[0]] and z[x[0]]
                if(!(t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1() && t_SMC0_0_s_0_n_2())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SMC0_1_s_0_n_0() {
                lock_ids[0] = target_locks[7] = 3; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3); // Check i
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SMC0_1_s_0_n_1() {
                lockManager.check_lock(3); // Check i
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | c[i]
            private boolean t_SMC0_1_s_0_n_2() {
                lock_ids[0] = target_locks[12] = 4 + i; // Acquire c[i]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3); // Check i
                lockManager.check_lock(4 + i); // Check c[i]
                if(c[i]) {
                    lock_ids[0] = target_locks[7]; // Release i
                    lock_ids[1] = target_locks[12]; // Release c[i]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lock_ids[1] = target_locks[12]; // Release c[i]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | i >= 0 and i < 2 and c[i]
            private boolean execute_transition_SMC0_1() {
                // SLCO expression | i >= 0 and i < 2 and c[i]
                if(!(t_SMC0_1_s_0_n_0() && t_SMC0_1_s_0_n_1() && t_SMC0_1_s_0_n_2())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | b[j]
            private boolean t_SMC0_2_s_0_n_0() {
                lock_ids[0] = target_locks[9] = 0; // Acquire j
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[11] = 1 + 0; // Acquire b[0]
                lock_ids[1] = target_locks[10] = 1 + 1; // Acquire b[1]
                lock_ids[2] = target_locks[8] = 1 + j; // Acquire b[j]
                lockManager.acquire_locks(lock_ids, 3);
                lockManager.check_lock(1 + j); // Check b[j]
                lockManager.check_lock(0); // Check j
                if(b[j]) {
                    lock_ids[0] = target_locks[8]; // Release b[j]
                    lock_ids[1] = target_locks[9]; // Release j
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[11]; // Release b[0]
                lock_ids[1] = target_locks[10]; // Release b[1]
                lock_ids[2] = target_locks[8]; // Release b[j]
                lock_ids[3] = target_locks[9]; // Release j
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO expression wrapper | i >= 0
            private boolean t_SMC0_2_s_0_n_1() {
                lock_ids[0] = target_locks[7] = 3; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3); // Check i
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lock_ids[1] = target_locks[10]; // Release b[1]
                lock_ids[2] = target_locks[11]; // Release b[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | i < 2
            private boolean t_SMC0_2_s_0_n_2() {
                lockManager.check_lock(3); // Check i
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lock_ids[1] = target_locks[10]; // Release b[1]
                lock_ids[2] = target_locks[11]; // Release b[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | b[i]
            private boolean t_SMC0_2_s_0_n_3() {
                lockManager.check_lock(3); // Check i
                lockManager.check_lock(1 + i); // Check b[i]
                if(b[i]) {
                    lock_ids[0] = target_locks[7]; // Release i
                    lock_ids[1] = target_locks[10]; // Release b[1]
                    lock_ids[2] = target_locks[11]; // Release b[0]
                    lockManager.release_locks(lock_ids, 3);
                    return true;
                }
                lock_ids[0] = target_locks[7]; // Release i
                lock_ids[1] = target_locks[10]; // Release b[1]
                lock_ids[2] = target_locks[11]; // Release b[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | b[j] and i >= 0 and i < 2 and b[i]
            private boolean execute_transition_SMC0_2() {
                // SLCO expression | b[j] and i >= 0 and i < 2 and b[i]
                if(!(t_SMC0_2_s_0_n_0() && t_SMC0_2_s_0_n_1() && t_SMC0_2_s_0_n_2() && t_SMC0_2_s_0_n_3())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            private void exec_SMC0() {
                // [N_DET.START]
                switch(random.nextInt(3)) {
                    case 0 -> {
                        // SLCO transition (id:0, p:0) | SMC0 -> SMC0 | guard: x[y[0]] and y[z[0]] and z[x[0]]
                        if(execute_transition_SMC0_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (id:1, p:0) | SMC0 -> SMC0 | guard: i >= 0 and i < 2 and c[i]
                        if(execute_transition_SMC0_1()) {
                            return;
                        }
                    }
                    case 2 -> {
                        // SLCO transition (id:2, p:0) | SMC0 -> SMC0 | guard: b[j] and i >= 0 and i < 2 and b[i]
                        if(execute_transition_SMC0_2()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            // Execute method
            private void exec() {
                while(true) {
                    lockManager.check_no_locks();
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

        T(boolean[] x, boolean[] y, boolean[] z, boolean[] b, boolean[] c, int i, int j) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(12);

            // Instantiate global variables
            this.x = x;
            this.y = y;
            this.z = z;
            this.b = b;
            this.c = c;
            this.i = i;
            this.j = j;

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

    // Representation of the SLCO class U
    private static class U implements SLCO_Class {
        // The threads
        private final Thread T_SM1;

        // Global variables
        private volatile int[] x;
        private volatile int[] y;
        private volatile int i;

        // Define the states fot the state machine SM1
        interface U_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1
            }
        }

        // Representation of the SLCO state machine SM1
        class SM1Thread extends Thread implements U_SM1Thread_States {
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
                lock_ids = new int[3];
                target_locks = new int[4];
                random = new Random();
            }

            // SLCO expression wrapper | x[0] > 0
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[2] = 1 + 1; // Acquire x[1]
                lock_ids[1] = target_locks[1] = 1 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 2);
                lockManager.check_lock(1 + 0); // Check x[0]
                if(x[0] > 0) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release x[1]
                lock_ids[1] = target_locks[1]; // Release x[0]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | y[0] >= 0
            private boolean t_SMC0_0_s_0_n_1() {
                lock_ids[0] = target_locks[3] = 3 + 0; // Acquire y[0]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3 + 0); // Check y[0]
                if(y[0] >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release x[1]
                lock_ids[1] = target_locks[1]; // Release x[0]
                lock_ids[2] = target_locks[3]; // Release y[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | y[0] < 2
            private boolean t_SMC0_0_s_0_n_2() {
                lockManager.check_lock(3 + 0); // Check y[0]
                if(y[0] < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release x[1]
                lock_ids[1] = target_locks[1]; // Release x[0]
                lock_ids[2] = target_locks[3]; // Release y[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | x[y[0]] > 0
            private boolean t_SMC0_0_s_0_n_3() {
                lockManager.check_lock(3 + 0); // Check y[0]
                lockManager.check_lock(1 + y[0]); // Check x[y[0]]
                if(x[y[0]] > 0) {
                    lock_ids[0] = target_locks[2]; // Release x[1]
                    lock_ids[1] = target_locks[1]; // Release x[0]
                    lock_ids[2] = target_locks[3]; // Release y[0]
                    lockManager.release_locks(lock_ids, 3);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release x[1]
                lock_ids[1] = target_locks[1]; // Release x[0]
                lock_ids[2] = target_locks[3]; // Release y[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | x[0] > 0 and y[0] >= 0 and y[0] < 2 and x[y[0]] > 0
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | x[0] > 0 and y[0] >= 0 and y[0] < 2 and x[y[0]] > 0
                if(!(t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1() && t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | y[x[0]] > 0
            private boolean t_SMC0_1_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 1 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 3 + x[0]; // Acquire y[x[0]]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3 + x[0]); // Check y[x[0]]
                lockManager.check_lock(1 + 0); // Check x[0]
                if(y[x[0]] > 0) {
                    lock_ids[0] = target_locks[0]; // Release y[x[0]]
                    lock_ids[1] = target_locks[1]; // Release x[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y[x[0]]
                lock_ids[1] = target_locks[1]; // Release x[0]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | y[x[0]] > 0
            private boolean execute_transition_SMC0_1() {
                // SLCO expression | y[x[0]] > 0
                if(!(t_SMC0_1_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | y[x[0]] > 0
            private boolean t_SMC0_2_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 1 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 3 + x[0]; // Acquire y[x[0]]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3 + x[0]); // Check y[x[0]]
                lockManager.check_lock(1 + 0); // Check x[0]
                if(y[x[0]] > 0) {
                    lock_ids[0] = target_locks[0]; // Release y[x[0]]
                    lock_ids[1] = target_locks[1]; // Release x[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y[x[0]]
                lock_ids[1] = target_locks[1]; // Release x[0]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | y[x[0]] > 0
            private boolean execute_transition_SMC0_2() {
                // SLCO expression | y[x[0]] > 0
                if(!(t_SMC0_2_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | y[x[0]] > 0
            private boolean t_SMC0_3_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 1 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 3 + x[0]; // Acquire y[x[0]]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3 + x[0]); // Check y[x[0]]
                lockManager.check_lock(1 + 0); // Check x[0]
                if(y[x[0]] > 0) {
                    lock_ids[0] = target_locks[0]; // Release y[x[0]]
                    lock_ids[1] = target_locks[1]; // Release x[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y[x[0]]
                lock_ids[1] = target_locks[1]; // Release x[0]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:3) | SMC0 -> SMC0 | y[x[0]] > 0
            private boolean execute_transition_SMC0_3() {
                // SLCO expression | y[x[0]] > 0
                if(!(t_SMC0_3_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | y[x[0]] > 0
            private boolean t_SMC0_4_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 1 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 3 + x[0]; // Acquire y[x[0]]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(3 + x[0]); // Check y[x[0]]
                lockManager.check_lock(1 + 0); // Check x[0]
                if(y[x[0]] > 0) {
                    lock_ids[0] = target_locks[0]; // Release y[x[0]]
                    lock_ids[1] = target_locks[1]; // Release x[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y[x[0]]
                lock_ids[1] = target_locks[1]; // Release x[0]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:4) | SMC0 -> SMC0 | y[x[0]] > 0
            private boolean execute_transition_SMC0_4() {
                // SLCO expression | y[x[0]] > 0
                if(!(t_SMC0_4_s_0_n_0())) {
                    return false;
                }

                currentState = SM1Thread.States.SMC0;
                return true;
            }

            private void exec_SMC0() {
                // [N_DET.START]
                switch(random.nextInt(2)) {
                    case 0 -> {
                        // SLCO transition (id:0, p:0) | SMC0 -> SMC0 | guard: x[0] > 0 and y[0] >= 0 and y[0] < 2 and x[y[0]] > 0
                        if(execute_transition_SMC0_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // [N_DET.START]
                        switch(random.nextInt(4)) {
                            case 0 -> {
                                // SLCO transition (id:1, p:0) | SMC0 -> SMC0 | guard: y[x[0]] > 0
                                if(execute_transition_SMC0_1()) {
                                    return;
                                }
                            }
                            case 1 -> {
                                // SLCO transition (id:2, p:0) | SMC0 -> SMC0 | guard: y[x[0]] > 0
                                if(execute_transition_SMC0_2()) {
                                    return;
                                }
                            }
                            case 2 -> {
                                // SLCO transition (id:3, p:0) | SMC0 -> SMC0 | guard: y[x[0]] > 0
                                if(execute_transition_SMC0_3()) {
                                    return;
                                }
                            }
                            case 3 -> {
                                // SLCO transition (id:4, p:0) | SMC0 -> SMC0 | guard: y[x[0]] > 0
                                if(execute_transition_SMC0_4()) {
                                    return;
                                }
                            }
                        }
                        // [N_DET.END]
                    }
                }
                // [N_DET.END]
            }

            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            // Execute method
            private void exec() {
                while(true) {
                    lockManager.check_no_locks();
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

        U(int[] x, int[] y, int i) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(5);

            // Instantiate global variables
            this.x = x;
            this.y = y;
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
                0,
                0
            ),
            new Q(
                new int[]{ 0, 0 },
                new int[]{ 0, 0 },
                0,
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