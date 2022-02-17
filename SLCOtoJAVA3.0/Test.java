import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

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
                0
            )
        };
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
        Test model = new Test();
        model.startThreads();
        model.joinThreads();
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

    // Representation of the SLCO class P.
    private static class P implements SLCO_Class {
        // The state machine threads.
        private final Thread T_SM1;
        private final Thread T_SM2;

        // Class variables.
        private final int[] x;
        private volatile int i;

        P(int[] x, int i) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(3);

            // Instantiate the class variables.
            this.x = x;
            this.i = i;

            // Instantiate the state machine threads and pass on the class' lock manager.
            T_SM1 = new P_SM1Thread(lockManager);
            T_SM2 = new P_SM2Thread(lockManager);
        }

        // Start all threads.
        public void startThreads() {
            T_SM1.start();
            T_SM2.start();
        }

        // Join all threads.
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

        // Define the states fot the state machine SM1.
        interface P_SM1Thread_States {
            enum States {
                SMC0, 
                SMC1
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
                lock_ids = new int[2];
                target_locks = new int[2];
                random = new Random();
            }

            // Main state machine loop.
            private void exec() {
                while(true) {
                    lockManager.check_no_locks();
                    switch(currentState) {
                        case SMC0 -> exec_SMC0();
                        case SMC1 -> exec_SMC1();
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

            // SLCO expression wrapper | i >= 0.
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check i
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
            }

            // SLCO expression wrapper | i < 2.
            private boolean t_SMC0_0_s_0_n_1() {
                lockManager.check_lock(0); // Check i
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
            }

            // SLCO expression wrapper | x[i] = 0.
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
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0.
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | i >= 0 and i < 2 and x[i] = 0.
                if(!(t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1() && t_SMC0_0_s_0_n_2())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // Attempt to fire a transition starting in state SMC0.
            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0.
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SMC1.
            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }
        }

        // Define the states fot the state machine SM2.
        interface P_SM2Thread_States {
            enum States {
                SMC0, 
                SMC1
            }
        }

        // Representation of the SLCO state machine SM2.
        class P_SM2Thread extends Thread implements P_SM2Thread_States {
            // Current state
            private P_SM2Thread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // Thread local variables.
            private int[] y;
            private int j;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            P_SM2Thread(LockManager lockManagerInstance) {
                currentState = P_SM2Thread.States.SMC0;
                lockManager = lockManagerInstance;
                lock_ids = new int[2];
                target_locks = new int[2];
                random = new Random();

                // Variable instantiations.
                y = new int[] { 0, 0 };
                j = 1;
            }

            // Main state machine loop.
            private void exec() {
                while(true) {
                    lockManager.check_no_locks();
                    switch(currentState) {
                        case SMC0 -> exec_SMC0();
                        case SMC1 -> exec_SMC1();
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

            // SLCO expression wrapper | i >= 0.
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check i
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
            }

            // SLCO expression wrapper | i < 2.
            private boolean t_SMC0_0_s_0_n_1() {
                lockManager.check_lock(0); // Check i
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
            }

            // SLCO expression wrapper | x[i] != 0.
            private boolean t_SMC0_0_s_0_n_2() {
                lock_ids[0] = target_locks[1] = 1 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                lockManager.check_lock(0); // Check i
                lockManager.check_lock(1 + i); // Check x[i]
                if(x[i] != 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[1]; // Release x[i]
                lockManager.release_locks(lock_ids, 2);
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
            private boolean execute_transition_SMC0_0() {
                // SLCO composite | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
                // SLCO expression | i >= 0 and i < 2 and x[i] != 0.
                if(!(t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1() && t_SMC0_0_s_0_n_2())) {
                    return false;
                }
                // SLCO assignment | x[i] := y[i].
                lockManager.check_lock(0); // Check i
                lockManager.check_lock(1 + i); // Check x[i]
                x[i] = y[i];
                lock_ids[0] = target_locks[1]; // Release x[i]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y[i] := 0.
                lockManager.check_lock(0); // Check i
                y[i] = 0;
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);

                currentState = P_SM2Thread.States.SMC0;
                return true;
            }

            // Attempt to fire a transition starting in state SMC0.
            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SMC1.
            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }
        }
    }
}