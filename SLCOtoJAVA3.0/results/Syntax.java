package testing;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.time.Duration;
import java.time.Instant;

// SLCO model Syntax.
public class Syntax {
    // The objects in the model.
    private final SLCO_Class[] objects;

    // Interface for SLCO classes.
    interface SLCO_Class {
        void startThreads();
        void joinThreads();
    }

    Syntax() {
        // Instantiate the objects.
        objects = new SLCO_Class[] {
            new P(
                False,
                False,
                False,
                False
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
        private volatile boolean a;
        private volatile boolean b;
        private volatile boolean c;
        private volatile boolean d;

        P(boolean a, boolean b, boolean c, boolean d) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(4);

            // Instantiate the class variables.
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;

            // Instantiate the state machine threads and pass on the class' lock manager.
            T_SM1 = new P_SM1Thread(lockManager);
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
                lock_ids = new int[1];
                target_locks = new int[4];
                random = new Random();
            }

            // SLCO expression wrapper | a.
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire a
                lockManager.acquire_locks(lock_ids, 1);
                if(a) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release a
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | b.
            private boolean t_SMC0_0_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 1; // Acquire b
                lockManager.acquire_locks(lock_ids, 1);
                if(b) {
                    lock_ids[0] = target_locks[1]; // Release b
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release b
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | c.
            private boolean t_SMC0_0_s_0_n_2() {
                lock_ids[0] = target_locks[2] = 2; // Acquire c
                lockManager.acquire_locks(lock_ids, 1);
                if(c) {
                    lock_ids[0] = target_locks[2]; // Release c
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release c
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | d.
            private boolean t_SMC0_0_s_0_n_3() {
                lock_ids[0] = target_locks[3] = 3; // Acquire d
                lockManager.acquire_locks(lock_ids, 1);
                if(d) {
                    lock_ids[0] = target_locks[3]; // Release d
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[3]; // Release d
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | a or b and c or d.
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | a or b and c or d.
                if(!(t_SMC0_0_s_0_n_0() || t_SMC0_0_s_0_n_1() && t_SMC0_0_s_0_n_2() || t_SMC0_0_s_0_n_3())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // Attempt to fire a transition starting in state SMC0.
            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | a or b and c or d.
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SMC1.
            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 60) {
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
        Syntax model = new Syntax();
        model.startThreads();
        model.joinThreads();
    }
}