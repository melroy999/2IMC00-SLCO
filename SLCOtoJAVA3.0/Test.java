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
        void lock(int[] lock_ids, int start, int end) {
            Arrays.sort(lock_ids, start, end);
            for (int i = start; i < end; i++) {
                locks[lock_ids[i]].lock();
            }
        }

        // Unlock method
        void unlock(int[] lock_ids, int start, int end) {
            for (int i = start; i < end; i++) {
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
                lock_ids = new int[0];
                target_locks = new int[0];
                random = new Random();
            }

            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 0; // Acquire i
                lockManager.acquire_locks(target_locks, 1);
                lock_ids[0] = target_locks[0] = 1 + 0; // Acquire x[0]
                lock_ids[1] = target_locks[2] = 1 + (i + 1); // Acquire x[(i + 1)]
                lockManager.acquire_locks(target_locks, 2);
                if(x[0] > 0) {
                    lock_ids[0] = target_locks[0]; // Release x[0]
                    lockManager.release_locks(target_locks, 1);
                    return True;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[1]; // Release i
                lock_ids[2] = target_locks[2]; // Release x[(i + 1)]
                lockManager.release_locks(target_locks, 3);
                return False;
            }

            private boolean t_SMC0_0_s_3_n_1() {
                lock_ids[0] = target_locks[1] = 0; // Acquire i
                lockManager.acquire_locks(target_locks, 1);
                lock_ids[0] = target_locks[0] = 1 + 0; // Acquire x[0]
                lockManager.acquire_locks(target_locks, 1);
                if(x[0] > 0) {
                    lock_ids[0] = target_locks[0]; // Release x[0]
                    lockManager.release_locks(target_locks, 1);
                    return True;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 2);
                return False;
            }

            private boolean t_SMC0_0_s_7_n_2() {
                lock_ids[0] = target_locks[1] = 0; // Acquire i
                lockManager.acquire_locks(target_locks, 1);
                lock_ids[0] = target_locks[0] = 1 + 0; // Acquire x[0]
                lockManager.acquire_locks(target_locks, 1);
                if(x[0] > 0) {
                    return True;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 2);
                return False;
            }

            private boolean t_SMC0_0_s_11_n_3() {
                lock_ids[0] = target_locks[1] = 0; // Acquire i
                lockManager.acquire_locks(target_locks, 1);
                if(i >= 0) {
                    return True;
                }
                lock_ids[0] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 1);
                return False;
            }

            private boolean t_SMC0_0_s_11_n_4() {
                if(i < 3) {
                    return True;
                }
                lock_ids[0] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 1);
                return False;
            }

            private boolean t_SMC0_0_s_11_n_5() {
                lock_ids[0] = target_locks[0] = 1 + 0; // Acquire x[0]
                lock_ids[1] = target_locks[3] = 1 + 1; // Acquire x[1]
                lockManager.acquire_locks(target_locks, 2);
                lock_ids[0] = target_locks[5] = 3 + i; // Acquire b[i]
                lockManager.acquire_locks(target_locks, 1);
                if(b[i]) {
                    lock_ids[0] = target_locks[5]; // Release b[i]
                    lockManager.release_locks(target_locks, 1);
                    return True;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[5]; // Release b[i]
                lock_ids[2] = target_locks[3]; // Release x[1]
                lock_ids[3] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 4);
                return False;
            }

            private boolean t_SMC0_0_s_15_n_6() {
                lock_ids[0] = target_locks[1] = 0; // Acquire i
                lockManager.acquire_locks(target_locks, 1);
                if(i >= 0) {
                    return True;
                }
                lock_ids[0] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 1);
                return False;
            }

            private boolean t_SMC0_0_s_15_n_7() {
                if(i < 2) {
                    return True;
                }
                lock_ids[0] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 1);
                return False;
            }

            private boolean t_SMC0_0_s_21_n_8() {
                if(i >= 0) {
                    return True;
                }
                lock_ids[0] = target_locks[9]; // Release b[0]
                lock_ids[1] = target_locks[7]; // Release b[1]
                lock_ids[2] = target_locks[8]; // Release b[2]
                lockManager.release_locks(target_locks, 3);
                return False;
            }

            private boolean t_SMC0_0_s_21_n_9() {
                if(i < 2) {
                    return True;
                }
                lock_ids[0] = target_locks[9]; // Release b[0]
                lock_ids[1] = target_locks[7]; // Release b[1]
                lock_ids[2] = target_locks[8]; // Release b[2]
                lockManager.release_locks(target_locks, 3);
                return False;
            }

            private boolean t_SMC0_0_s_21_n_10() {
                if(b[i]) {
                    lock_ids[0] = target_locks[9]; // Release b[0]
                    lock_ids[1] = target_locks[7]; // Release b[1]
                    lock_ids[2] = target_locks[8]; // Release b[2]
                    lockManager.release_locks(target_locks, 3);
                    return True;
                }
                lock_ids[0] = target_locks[9]; // Release b[0]
                lock_ids[1] = target_locks[7]; // Release b[1]
                lock_ids[2] = target_locks[8]; // Release b[2]
                lockManager.release_locks(target_locks, 3);
                return False;
            }

            // Representation of SLCO transition SMC0_0 (SMC0 -> SMC0)
            private boolean execute_transition_SMC0_0() {
                // SLCO statement: [x[0] > 0; i := i + 1; x[i] := 1] -> [x[0] > 0; i := i + 1; x[i] := 1]
                if(!(t_SMC0_0_s_0_n_0())) {
                    return False;
                }
                i = i + 1;
                x[i] = 1;
                lock_ids[0] = target_locks[2]; // Release x[(i + 1)]
                lock_ids[1] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 2);

                // SLCO statement: [x[0] > 0; i := 2; x[i] := 1; x[1] := 1] -> [x[0] > 0; i := 2; x[i] := 1; x[1] := 1]
                if(!(t_SMC0_0_s_3_n_1())) {
                    return False;
                }
                lock_ids[0] = target_locks[3] = 1 + 1; // Acquire x[1]
                lock_ids[1] = target_locks[4] = 1 + 2; // Acquire x[2]
                lockManager.acquire_locks(target_locks, 2);
                i = 2;
                x[i] = 1;
                lock_ids[0] = target_locks[4]; // Release x[2]
                lock_ids[1] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 2);
                x[1] = 1;
                lock_ids[0] = target_locks[3]; // Release x[1]
                lockManager.release_locks(target_locks, 1);

                // SLCO statement: [x[0] > 0; i := 0; x[i] := 1; x[1] := 1] -> [x[0] > 0; i := 0; x[i] := 1; x[1] := 1]
                if(!(t_SMC0_0_s_7_n_2())) {
                    return False;
                }
                i = 0;
                lock_ids[0] = target_locks[3] = 1 + 1; // Acquire x[1]
                lockManager.acquire_locks(target_locks, 1);
                x[i] = 1;
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 2);
                x[1] = 1;
                lock_ids[0] = target_locks[3]; // Release x[1]
                lockManager.release_locks(target_locks, 1);

                // SLCO statement: [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1] -> [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1]
                if(!(t_SMC0_0_s_11_n_3() && t_SMC0_0_s_11_n_4() && t_SMC0_0_s_11_n_5())) {
                    return False;
                }
                i = 0;
                x[i] = 1;
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 2);
                x[1] = 1;
                lock_ids[0] = target_locks[3]; // Release x[1]
                lockManager.release_locks(target_locks, 1);

                // SLCO statement: [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0] -> [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0]
                if(!(t_SMC0_0_s_15_n_6() && t_SMC0_0_s_15_n_7())) {
                    return False;
                }
                i = 0;
                lock_ids[0] = target_locks[6] = 1 + i; // Acquire x[i]
                lock_ids[1] = target_locks[0] = 1 + 0; // Acquire x[0]
                lockManager.acquire_locks(target_locks, 2);
                x[i] = 0;
                x[x[i]] = 0;
                lock_ids[0] = target_locks[6]; // Release x[i]
                lock_ids[1] = target_locks[0]; // Release x[0]
                lock_ids[2] = target_locks[1]; // Release i
                lockManager.release_locks(target_locks, 3);

                // SLCO statement: [i := 0; b[i] := i >= 0 and i < 2 and b[i]] -> [true; i := 0; b[i] := i >= 0 and i < 2 and b[i]]
                if(!(true)) {
                    return False;
                }
                lock_ids[0] = target_locks[1] = 0; // Acquire i
                lockManager.acquire_locks(target_locks, 1);
                i = 0;
                lock_ids[0] = target_locks[8] = 3 + 2; // Acquire b[2]
                lock_ids[1] = target_locks[9] = 3 + 0; // Acquire b[0]
                lock_ids[2] = target_locks[5] = 3 + i; // Acquire b[i]
                lock_ids[3] = target_locks[7] = 3 + 1; // Acquire b[1]
                lockManager.acquire_locks(target_locks, 4);
                b[i] = t_SMC0_0_s_21_n_8() && t_SMC0_0_s_21_n_9() && t_SMC0_0_s_21_n_10();
                lock_ids[0] = target_locks[1]; // Release i
                lock_ids[1] = target_locks[5]; // Release b[i]
                lockManager.release_locks(target_locks, 2);

                // SLCO statement: [b[0] := 0; i := 0; b[i] := i >= 0 and i < 2 and b[i]] -> [true; b[0] := 0; i := 0; b[i] := i >= 0 and i < 2 and b[i]]
                if(!(true)) {
                    return False;
                }
                lock_ids[0] = target_locks[1] = 0; // Acquire i
                lockManager.acquire_locks(target_locks, 1);
                lock_ids[0] = target_locks[9] = 3 + 0; // Acquire b[0]
                lockManager.acquire_locks(target_locks, 1);
                b[0] = 0;
                i = 0;
                b[i] = i >= 0 && i < 2 && b[i];
                lock_ids[0] = target_locks[1]; // Release i
                lock_ids[1] = target_locks[9]; // Release b[0]
                lockManager.release_locks(target_locks, 2);

                return true;
            }

            private void exec_SMC0() {
                if(execute_transition_SMC0_0()) {
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

        P(int[] x, int[] y, boolean[] b, int i) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(1);

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
                new int[]{0, 0},
                new int[]{0, 0},
                new boolean[]{False, False, False},
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