package testing;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.time.Duration;
import java.time.Instant;

// SLCO model ToadsAndFrogs.
public class ToadsAndFrogs {
    // The objects in the model.
    private final SLCO_Class[] objects;

    // Interface for SLCO classes.
    interface SLCO_Class {
        void startThreads();
        void joinThreads();
    }

    ToadsAndFrogs() {
        // Instantiate the objects.
        objects = new SLCO_Class[] {
            new GlobalClass(
                new int[]{ 1, 1, 1, 1, 0, 2, 2, 2, 2 },
                4,
                0,
                8
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

    // Representation of the SLCO class GlobalClass.
    private static class GlobalClass implements SLCO_Class {
        // The state machine threads.
        private final Thread T_Toad;
        private final Thread T_Frog;
        private final Thread T_Check;

        // Class variables.
        private final int[] a;
        private volatile int y;
        private volatile int tmin;
        private volatile int fmax;

        GlobalClass(int[] a, int y, int tmin, int fmax) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(12);

            // Instantiate the class variables.
            this.a = a;
            this.y = y;
            this.tmin = tmin;
            this.fmax = fmax;

            // Instantiate the state machine threads and pass on the class' lock manager.
            T_Toad = new GlobalClass_ToadThread(lockManager);
            T_Frog = new GlobalClass_FrogThread(lockManager);
            T_Check = new GlobalClass_CheckThread(lockManager);
        }

        // Define the states fot the state machine Toad.
        interface GlobalClass_ToadThread_States {
            enum States {
                q
            }
        }

        // Representation of the SLCO state machine Toad.
        class GlobalClass_ToadThread extends Thread implements GlobalClass_ToadThread_States {
            // Current state
            private GlobalClass_ToadThread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            GlobalClass_ToadThread(LockManager lockManagerInstance) {
                currentState = GlobalClass_ToadThread.States.q;
                lockManager = lockManagerInstance;
                lock_ids = new int[5];
                target_locks = new int[5];
                random = new Random();
            }

            // SLCO expression wrapper | y > 0.
            private boolean t_q_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y > 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | tmin != y - 1.
            private boolean t_q_0_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 1; // Acquire tmin
                lockManager.acquire_locks(lock_ids, 1);
                if(tmin != y - 1) {
                    lock_ids[0] = target_locks[1]; // Release tmin
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 1.
            private boolean t_q_0_s_0_n_2() {
                lock_ids[0] = target_locks[2] = 3 + (y - 1); // Acquire a[(y - 1)]
                lock_ids[1] = target_locks[3] = 3 + y; // Acquire a[y]
                lockManager.acquire_locks(lock_ids, 2);
                if(a[y - 1] == 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[(y - 1)]
                lock_ids[2] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO transition (p:0, id:0) | q -> q | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
            private boolean execute_transition_q_0() {
                // SLCO composite | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
                // SLCO expression | y > 0 and tmin != y - 1 and a[y - 1] = 1.
                if(!(t_q_0_s_0_n_0() && t_q_0_s_0_n_1() && t_q_0_s_0_n_2())) {
                    return false;
                }
                // SLCO assignment | a[y] := 1.
                a[y] = 1;
                lock_ids[0] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y - 1.
                y = y - 1;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[(y - 1)]
                lockManager.release_locks(lock_ids, 2);

                currentState = GlobalClass_ToadThread.States.q;
                return true;
            }

            // SLCO expression wrapper | y > 0.
            private boolean t_q_1_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y > 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | tmin = y - 1.
            private boolean t_q_1_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 1; // Acquire tmin
                lockManager.acquire_locks(lock_ids, 1);
                if(tmin == y - 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 1.
            private boolean t_q_1_s_0_n_2() {
                lock_ids[0] = target_locks[2] = 3 + (y - 1); // Acquire a[(y - 1)]
                lock_ids[1] = target_locks[3] = 3 + y; // Acquire a[y]
                lockManager.acquire_locks(lock_ids, 2);
                if(a[y - 1] == 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lock_ids[2] = target_locks[2]; // Release a[(y - 1)]
                lock_ids[3] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO transition (p:0, id:1) | q -> q | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
            private boolean execute_transition_q_1() {
                // SLCO composite | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
                // SLCO expression | y > 0 and tmin = y - 1 and a[y - 1] = 1.
                if(!(t_q_1_s_0_n_0() && t_q_1_s_0_n_1() && t_q_1_s_0_n_2())) {
                    return false;
                }
                // SLCO assignment | a[y] := 1.
                a[y] = 1;
                lock_ids[0] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | tmin := y.
                tmin = y;
                lock_ids[0] = target_locks[1]; // Release tmin
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y - 1.
                y = y - 1;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[(y - 1)]
                lockManager.release_locks(lock_ids, 2);

                currentState = GlobalClass_ToadThread.States.q;
                return true;
            }

            // SLCO expression wrapper | y > 1.
            private boolean t_q_2_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y > 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | tmin != y - 2.
            private boolean t_q_2_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 1; // Acquire tmin
                lockManager.acquire_locks(lock_ids, 1);
                if(tmin != y - 2) {
                    lock_ids[0] = target_locks[1]; // Release tmin
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | a[y - 2] = 1.
            private boolean t_q_2_s_0_n_2() {
                lock_ids[0] = target_locks[2] = 3 + (y - 1); // Acquire a[(y - 1)]
                lock_ids[1] = target_locks[3] = 3 + y; // Acquire a[y]
                lock_ids[2] = target_locks[4] = 3 + (y - 2); // Acquire a[(y - 2)]
                lockManager.acquire_locks(lock_ids, 3);
                if(a[y - 2] == 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[(y - 1)]
                lock_ids[2] = target_locks[3]; // Release a[y]
                lock_ids[3] = target_locks[4]; // Release a[(y - 2)]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 2.
            private boolean t_q_2_s_0_n_3() {
                if(a[y - 1] == 2) {
                    lock_ids[0] = target_locks[2]; // Release a[(y - 1)]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[(y - 1)]
                lock_ids[2] = target_locks[3]; // Release a[y]
                lock_ids[3] = target_locks[4]; // Release a[(y - 2)]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO transition (p:0, id:2) | q -> q | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
            private boolean execute_transition_q_2() {
                // SLCO composite | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
                // SLCO expression | y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
                if(!(t_q_2_s_0_n_0() && t_q_2_s_0_n_1() && t_q_2_s_0_n_2() && t_q_2_s_0_n_3())) {
                    return false;
                }
                // SLCO assignment | a[y] := 1.
                a[y] = 1;
                lock_ids[0] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y - 2.
                y = y - 2;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[4]; // Release a[(y - 2)]
                lockManager.release_locks(lock_ids, 2);

                currentState = GlobalClass_ToadThread.States.q;
                return true;
            }

            // SLCO expression wrapper | y > 1.
            private boolean t_q_3_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y > 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | tmin = y - 2.
            private boolean t_q_3_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 1; // Acquire tmin
                lockManager.acquire_locks(lock_ids, 1);
                if(tmin == y - 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | a[y - 2] = 1.
            private boolean t_q_3_s_0_n_2() {
                lock_ids[0] = target_locks[2] = 3 + (y - 1); // Acquire a[(y - 1)]
                lock_ids[1] = target_locks[3] = 3 + y; // Acquire a[y]
                lock_ids[2] = target_locks[4] = 3 + (y - 2); // Acquire a[(y - 2)]
                lockManager.acquire_locks(lock_ids, 3);
                if(a[y - 2] == 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lock_ids[2] = target_locks[2]; // Release a[(y - 1)]
                lock_ids[3] = target_locks[3]; // Release a[y]
                lock_ids[4] = target_locks[4]; // Release a[(y - 2)]
                lockManager.release_locks(lock_ids, 5);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 2.
            private boolean t_q_3_s_0_n_3() {
                if(a[y - 1] == 2) {
                    lock_ids[0] = target_locks[2]; // Release a[(y - 1)]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lock_ids[2] = target_locks[2]; // Release a[(y - 1)]
                lock_ids[3] = target_locks[3]; // Release a[y]
                lock_ids[4] = target_locks[4]; // Release a[(y - 2)]
                lockManager.release_locks(lock_ids, 5);
                return false;
            }

            // SLCO transition (p:0, id:3) | q -> q | [y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; tmin := y; y := y - 2; a[y] := 0].
            private boolean execute_transition_q_3() {
                // SLCO composite | [y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; tmin := y; y := y - 2; a[y] := 0].
                // SLCO expression | y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
                if(!(t_q_3_s_0_n_0() && t_q_3_s_0_n_1() && t_q_3_s_0_n_2() && t_q_3_s_0_n_3())) {
                    return false;
                }
                // SLCO assignment | a[y] := 1.
                a[y] = 1;
                lock_ids[0] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | tmin := y.
                tmin = y;
                lock_ids[0] = target_locks[1]; // Release tmin
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y - 2.
                y = y - 2;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[4]; // Release a[(y - 2)]
                lockManager.release_locks(lock_ids, 2);

                currentState = GlobalClass_ToadThread.States.q;
                return true;
            }

            // Attempt to fire a transition starting in state q.
            private void exec_q() {
                // [N_DET.START]
                switch(random.nextInt(4)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | q -> q | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
                        if(execute_transition_q_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | q -> q | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
                        if(execute_transition_q_1()) {
                            return;
                        }
                    }
                    case 2 -> {
                        // SLCO transition (p:0, id:2) | q -> q | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
                        if(execute_transition_q_2()) {
                            return;
                        }
                    }
                    case 3 -> {
                        // SLCO transition (p:0, id:3) | q -> q | [y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; tmin := y; y := y - 2; a[y] := 0].
                        if(execute_transition_q_3()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 60) {
                    switch(currentState) {
                        case q -> exec_q();
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

        // Define the states fot the state machine Frog.
        interface GlobalClass_FrogThread_States {
            enum States {
                q
            }
        }

        // Representation of the SLCO state machine Frog.
        class GlobalClass_FrogThread extends Thread implements GlobalClass_FrogThread_States {
            // Current state
            private GlobalClass_FrogThread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            GlobalClass_FrogThread(LockManager lockManagerInstance) {
                currentState = GlobalClass_FrogThread.States.q;
                lockManager = lockManagerInstance;
                lock_ids = new int[5];
                target_locks = new int[5];
                random = new Random();
            }

            // SLCO expression wrapper | y < 8.
            private boolean t_q_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y < 8) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | fmax != y + 1.
            private boolean t_q_0_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 2; // Acquire fmax
                lockManager.acquire_locks(lock_ids, 1);
                if(fmax != y + 1) {
                    lock_ids[0] = target_locks[1]; // Release fmax
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 2.
            private boolean t_q_0_s_0_n_2() {
                lock_ids[0] = target_locks[2] = 3 + (y + 1); // Acquire a[(y + 1)]
                lock_ids[1] = target_locks[3] = 3 + y; // Acquire a[y]
                lockManager.acquire_locks(lock_ids, 2);
                if(a[y + 1] == 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[(y + 1)]
                lock_ids[2] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO transition (p:0, id:0) | q -> q | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
            private boolean execute_transition_q_0() {
                // SLCO composite | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
                // SLCO expression | y < 8 and fmax != y + 1 and a[y + 1] = 2.
                if(!(t_q_0_s_0_n_0() && t_q_0_s_0_n_1() && t_q_0_s_0_n_2())) {
                    return false;
                }
                // SLCO assignment | a[y] := 2.
                a[y] = 2;
                lock_ids[0] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y + 1.
                y = y + 1;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[(y + 1)]
                lockManager.release_locks(lock_ids, 2);

                currentState = GlobalClass_FrogThread.States.q;
                return true;
            }

            // SLCO expression wrapper | y < 8.
            private boolean t_q_1_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y < 8) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | fmax = y + 1.
            private boolean t_q_1_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 2; // Acquire fmax
                lockManager.acquire_locks(lock_ids, 1);
                if(fmax == y + 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 2.
            private boolean t_q_1_s_0_n_2() {
                lock_ids[0] = target_locks[2] = 3 + (y + 1); // Acquire a[(y + 1)]
                lock_ids[1] = target_locks[3] = 3 + y; // Acquire a[y]
                lockManager.acquire_locks(lock_ids, 2);
                if(a[y + 1] == 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lock_ids[2] = target_locks[2]; // Release a[(y + 1)]
                lock_ids[3] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO transition (p:0, id:1) | q -> q | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
            private boolean execute_transition_q_1() {
                // SLCO composite | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
                // SLCO expression | y < 8 and fmax = y + 1 and a[y + 1] = 2.
                if(!(t_q_1_s_0_n_0() && t_q_1_s_0_n_1() && t_q_1_s_0_n_2())) {
                    return false;
                }
                // SLCO assignment | a[y] := 2.
                a[y] = 2;
                lock_ids[0] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | fmax := y.
                fmax = y;
                lock_ids[0] = target_locks[1]; // Release fmax
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y + 1.
                y = y + 1;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[(y + 1)]
                lockManager.release_locks(lock_ids, 2);

                currentState = GlobalClass_FrogThread.States.q;
                return true;
            }

            // SLCO expression wrapper | y < 7.
            private boolean t_q_2_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y < 7) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | fmax != y + 2.
            private boolean t_q_2_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 2; // Acquire fmax
                lockManager.acquire_locks(lock_ids, 1);
                if(fmax != y + 2) {
                    lock_ids[0] = target_locks[1]; // Release fmax
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 1.
            private boolean t_q_2_s_0_n_2() {
                lock_ids[0] = target_locks[2] = 3 + (y + 1); // Acquire a[(y + 1)]
                lock_ids[1] = target_locks[3] = 3 + y; // Acquire a[y]
                lock_ids[2] = target_locks[4] = 3 + y + 2; // Acquire a[y + 2]
                lockManager.acquire_locks(lock_ids, 3);
                if(a[y + 1] == 1) {
                    lock_ids[0] = target_locks[2]; // Release a[(y + 1)]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[(y + 1)]
                lock_ids[2] = target_locks[3]; // Release a[y]
                lock_ids[3] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO expression wrapper | a[y + 2] = 2.
            private boolean t_q_2_s_0_n_3() {
                if(a[y + 2] == 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[3]; // Release a[y]
                lock_ids[2] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO transition (p:0, id:2) | q -> q | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
            private boolean execute_transition_q_2() {
                // SLCO composite | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
                // SLCO expression | y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
                if(!(t_q_2_s_0_n_0() && t_q_2_s_0_n_1() && t_q_2_s_0_n_2() && t_q_2_s_0_n_3())) {
                    return false;
                }
                // SLCO assignment | a[y] := 2.
                a[y] = 2;
                lock_ids[0] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y + 2.
                y = y + 2;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 2);

                currentState = GlobalClass_FrogThread.States.q;
                return true;
            }

            // SLCO expression wrapper | y < 7.
            private boolean t_q_3_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y < 7) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | fmax = y + 2.
            private boolean t_q_3_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 2; // Acquire fmax
                lockManager.acquire_locks(lock_ids, 1);
                if(fmax == y + 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 1.
            private boolean t_q_3_s_0_n_2() {
                lock_ids[0] = target_locks[2] = 3 + (y + 1); // Acquire a[(y + 1)]
                lock_ids[1] = target_locks[3] = 3 + y; // Acquire a[y]
                lock_ids[2] = target_locks[4] = 3 + y + 2; // Acquire a[y + 2]
                lockManager.acquire_locks(lock_ids, 3);
                if(a[y + 1] == 1) {
                    lock_ids[0] = target_locks[2]; // Release a[(y + 1)]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lock_ids[2] = target_locks[2]; // Release a[(y + 1)]
                lock_ids[3] = target_locks[3]; // Release a[y]
                lock_ids[4] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 5);
                return false;
            }

            // SLCO expression wrapper | a[y + 2] = 2.
            private boolean t_q_3_s_0_n_3() {
                if(a[y + 2] == 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lock_ids[2] = target_locks[3]; // Release a[y]
                lock_ids[3] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO transition (p:0, id:3) | q -> q | [y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; fmax := y; y := y + 2; a[y] := 0].
            private boolean execute_transition_q_3() {
                // SLCO composite | [y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; fmax := y; y := y + 2; a[y] := 0].
                // SLCO expression | y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
                if(!(t_q_3_s_0_n_0() && t_q_3_s_0_n_1() && t_q_3_s_0_n_2() && t_q_3_s_0_n_3())) {
                    return false;
                }
                // SLCO assignment | a[y] := 2.
                a[y] = 2;
                lock_ids[0] = target_locks[3]; // Release a[y]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | fmax := y.
                fmax = y;
                lock_ids[0] = target_locks[1]; // Release fmax
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y + 2.
                y = y + 2;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 2);

                currentState = GlobalClass_FrogThread.States.q;
                return true;
            }

            // Attempt to fire a transition starting in state q.
            private void exec_q() {
                // [N_DET.START]
                switch(random.nextInt(4)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | q -> q | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
                        if(execute_transition_q_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | q -> q | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
                        if(execute_transition_q_1()) {
                            return;
                        }
                    }
                    case 2 -> {
                        // SLCO transition (p:0, id:2) | q -> q | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
                        if(execute_transition_q_2()) {
                            return;
                        }
                    }
                    case 3 -> {
                        // SLCO transition (p:0, id:3) | q -> q | [y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; fmax := y; y := y + 2; a[y] := 0].
                        if(execute_transition_q_3()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 60) {
                    switch(currentState) {
                        case q -> exec_q();
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

        // Define the states fot the state machine Check.
        interface GlobalClass_CheckThread_States {
            enum States {
                running, 
                success, 
                failure, 
                reset
            }
        }

        // Representation of the SLCO state machine Check.
        class GlobalClass_CheckThread extends Thread implements GlobalClass_CheckThread_States {
            // Current state
            private GlobalClass_CheckThread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            GlobalClass_CheckThread(LockManager lockManagerInstance) {
                currentState = GlobalClass_CheckThread.States.running;
                lockManager = lockManagerInstance;
                lock_ids = new int[5];
                target_locks = new int[12];
                random = new Random();
            }

            // SLCO expression wrapper | tmin > y.
            private boolean t_running_0_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 1; // Acquire tmin
                lockManager.acquire_locks(lock_ids, 1);
                if(tmin > y) {
                    lock_ids[0] = target_locks[0]; // Release tmin
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release tmin
                lock_ids[1] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | fmax < y.
            private boolean t_running_0_s_0_n_1() {
                lock_ids[0] = target_locks[2] = 2; // Acquire fmax
                lockManager.acquire_locks(lock_ids, 1);
                if(fmax < y) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lock_ids[1] = target_locks[2]; // Release fmax
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[2]; // Release fmax
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | running -> success | tmin > y and fmax < y.
            private boolean execute_transition_running_0() {
                // SLCO expression | tmin > y and fmax < y.
                if(!(t_running_0_s_0_n_0() && t_running_0_s_0_n_1())) {
                    return false;
                }

                currentState = GlobalClass_CheckThread.States.success;
                return true;
            }

            // SLCO expression wrapper | y = 0.
            private boolean t_running_1_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y == 0) {
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 1.
            private boolean t_running_1_s_0_n_1() {
                lock_ids[0] = target_locks[3] = 3 + y + 1; // Acquire a[y + 1]
                lock_ids[1] = target_locks[4] = 3 + y + 2; // Acquire a[y + 2]
                lockManager.acquire_locks(lock_ids, 2);
                if(a[y + 1] == 1) {
                    lock_ids[0] = target_locks[3]; // Release a[y + 1]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[3]; // Release a[y + 1]
                lock_ids[2] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | a[y + 2] = 1.
            private boolean t_running_1_s_0_n_2() {
                if(a[y + 2] == 1) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lock_ids[1] = target_locks[4]; // Release a[y + 2]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:1) | running -> failure | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
            private boolean execute_transition_running_1() {
                // SLCO expression | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
                if(!(t_running_1_s_0_n_0() && t_running_1_s_0_n_1() && t_running_1_s_0_n_2())) {
                    return false;
                }

                currentState = GlobalClass_CheckThread.States.failure;
                return true;
            }

            // SLCO expression wrapper | y = 1.
            private boolean t_running_2_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y == 1) {
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 2.
            private boolean t_running_2_s_0_n_1() {
                lock_ids[0] = target_locks[3] = 3 + y + 1; // Acquire a[y + 1]
                lock_ids[1] = target_locks[4] = 3 + y + 2; // Acquire a[y + 2]
                lock_ids[2] = target_locks[6] = 3 + y - 1; // Acquire a[y - 1]
                lockManager.acquire_locks(lock_ids, 3);
                if(a[y - 1] == 2) {
                    lock_ids[0] = target_locks[6]; // Release a[y - 1]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[3]; // Release a[y + 1]
                lock_ids[2] = target_locks[4]; // Release a[y + 2]
                lock_ids[3] = target_locks[6]; // Release a[y - 1]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 1.
            private boolean t_running_2_s_0_n_2() {
                if(a[y + 1] == 1) {
                    lock_ids[0] = target_locks[3]; // Release a[y + 1]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[3]; // Release a[y + 1]
                lock_ids[2] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | a[y + 2] = 1.
            private boolean t_running_2_s_0_n_3() {
                if(a[y + 2] == 1) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lock_ids[1] = target_locks[4]; // Release a[y + 2]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:2) | running -> failure | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
            private boolean execute_transition_running_2() {
                // SLCO expression | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
                if(!(t_running_2_s_0_n_0() && t_running_2_s_0_n_1() && t_running_2_s_0_n_2() && t_running_2_s_0_n_3())) {
                    return false;
                }

                currentState = GlobalClass_CheckThread.States.failure;
                return true;
            }

            // SLCO expression wrapper | y > 1.
            private boolean t_running_3_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y > 1) {
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | y < 7.
            private boolean t_running_3_s_0_n_1() {
                if(y < 7) {
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | a[y - 2] = 2.
            private boolean t_running_3_s_0_n_2() {
                lock_ids[0] = target_locks[3] = 3 + y + 1; // Acquire a[y + 1]
                lock_ids[1] = target_locks[4] = 3 + y + 2; // Acquire a[y + 2]
                lock_ids[2] = target_locks[5] = 3 + y - 2; // Acquire a[y - 2]
                lock_ids[3] = target_locks[6] = 3 + y - 1; // Acquire a[y - 1]
                lockManager.acquire_locks(lock_ids, 4);
                if(a[y - 2] == 2) {
                    lock_ids[0] = target_locks[5]; // Release a[y - 2]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[3]; // Release a[y + 1]
                lock_ids[2] = target_locks[4]; // Release a[y + 2]
                lock_ids[3] = target_locks[5]; // Release a[y - 2]
                lock_ids[4] = target_locks[6]; // Release a[y - 1]
                lockManager.release_locks(lock_ids, 5);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 2.
            private boolean t_running_3_s_0_n_3() {
                if(a[y - 1] == 2) {
                    lock_ids[0] = target_locks[6]; // Release a[y - 1]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[3]; // Release a[y + 1]
                lock_ids[2] = target_locks[4]; // Release a[y + 2]
                lock_ids[3] = target_locks[6]; // Release a[y - 1]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 1.
            private boolean t_running_3_s_0_n_4() {
                if(a[y + 1] == 1) {
                    lock_ids[0] = target_locks[3]; // Release a[y + 1]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[3]; // Release a[y + 1]
                lock_ids[2] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | a[y + 2] = 1.
            private boolean t_running_3_s_0_n_5() {
                if(a[y + 2] == 1) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lock_ids[1] = target_locks[4]; // Release a[y + 2]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[4]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:3) | running -> failure | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
            private boolean execute_transition_running_3() {
                // SLCO expression | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
                if(!(t_running_3_s_0_n_0() && t_running_3_s_0_n_1() && t_running_3_s_0_n_2() && t_running_3_s_0_n_3() && t_running_3_s_0_n_4() && t_running_3_s_0_n_5())) {
                    return false;
                }

                currentState = GlobalClass_CheckThread.States.failure;
                return true;
            }

            // SLCO expression wrapper | y = 7.
            private boolean t_running_4_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y == 7) {
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | a[y - 2] = 2.
            private boolean t_running_4_s_0_n_1() {
                lock_ids[0] = target_locks[3] = 3 + y + 1; // Acquire a[y + 1]
                lock_ids[1] = target_locks[5] = 3 + y - 2; // Acquire a[y - 2]
                lock_ids[2] = target_locks[6] = 3 + y - 1; // Acquire a[y - 1]
                lockManager.acquire_locks(lock_ids, 3);
                if(a[y - 2] == 2) {
                    lock_ids[0] = target_locks[5]; // Release a[y - 2]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[3]; // Release a[y + 1]
                lock_ids[2] = target_locks[5]; // Release a[y - 2]
                lock_ids[3] = target_locks[6]; // Release a[y - 1]
                lockManager.release_locks(lock_ids, 4);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 2.
            private boolean t_running_4_s_0_n_2() {
                if(a[y - 1] == 2) {
                    lock_ids[0] = target_locks[6]; // Release a[y - 1]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[3]; // Release a[y + 1]
                lock_ids[2] = target_locks[6]; // Release a[y - 1]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 1.
            private boolean t_running_4_s_0_n_3() {
                if(a[y + 1] == 1) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lock_ids[1] = target_locks[3]; // Release a[y + 1]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[3]; // Release a[y + 1]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:4) | running -> failure | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
            private boolean execute_transition_running_4() {
                // SLCO expression | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
                if(!(t_running_4_s_0_n_0() && t_running_4_s_0_n_1() && t_running_4_s_0_n_2() && t_running_4_s_0_n_3())) {
                    return false;
                }

                currentState = GlobalClass_CheckThread.States.failure;
                return true;
            }

            // SLCO expression wrapper | y = 8.
            private boolean t_running_5_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y == 8) {
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | a[y - 2] = 2.
            private boolean t_running_5_s_0_n_1() {
                lock_ids[0] = target_locks[5] = 3 + y - 2; // Acquire a[y - 2]
                lock_ids[1] = target_locks[6] = 3 + y - 1; // Acquire a[y - 1]
                lockManager.acquire_locks(lock_ids, 2);
                if(a[y - 2] == 2) {
                    lock_ids[0] = target_locks[5]; // Release a[y - 2]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[5]; // Release a[y - 2]
                lock_ids[2] = target_locks[6]; // Release a[y - 1]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 2.
            private boolean t_running_5_s_0_n_2() {
                if(a[y - 1] == 2) {
                    lock_ids[0] = target_locks[1]; // Release y
                    lock_ids[1] = target_locks[6]; // Release a[y - 1]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release y
                lock_ids[1] = target_locks[6]; // Release a[y - 1]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:5) | running -> failure | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
            private boolean execute_transition_running_5() {
                // SLCO expression | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
                if(!(t_running_5_s_0_n_0() && t_running_5_s_0_n_1() && t_running_5_s_0_n_2())) {
                    return false;
                }

                currentState = GlobalClass_CheckThread.States.failure;
                return true;
            }

            // SLCO transition (p:0, id:0) | success -> reset | true.
            private boolean execute_transition_success_0() {
                // (Superfluous) SLCO expression | true.

                currentState = GlobalClass_CheckThread.States.reset;
                return true;
            }

            // SLCO transition (p:0, id:0) | failure -> reset | true.
            private boolean execute_transition_failure_0() {
                // (Superfluous) SLCO expression | true.

                currentState = GlobalClass_CheckThread.States.reset;
                return true;
            }

            // SLCO transition (p:0, id:0) | reset -> running | true | [true; y := 4; tmin := 0; fmax := 8; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[4] := 0; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
            private boolean execute_transition_reset_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [y := 4; tmin := 0; fmax := 8; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[4] := 0; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2] -> [true; y := 4; tmin := 0; fmax := 8; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[4] := 0; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | y := 4.
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                y = 4;
                lock_ids[0] = target_locks[0]; // Release y
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | tmin := 0.
                lock_ids[0] = target_locks[1] = 1; // Acquire tmin
                lockManager.acquire_locks(lock_ids, 1);
                tmin = 0;
                lock_ids[0] = target_locks[1]; // Release tmin
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | fmax := 8.
                lock_ids[0] = target_locks[2] = 2; // Acquire fmax
                lockManager.acquire_locks(lock_ids, 1);
                fmax = 8;
                lock_ids[0] = target_locks[2]; // Release fmax
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[0] := 1.
                lock_ids[0] = target_locks[3] = 3 + 0; // Acquire a[0]
                lockManager.acquire_locks(lock_ids, 1);
                a[0] = 1;
                lock_ids[0] = target_locks[3]; // Release a[0]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[1] := 1.
                lock_ids[0] = target_locks[4] = 3 + 1; // Acquire a[1]
                lockManager.acquire_locks(lock_ids, 1);
                a[1] = 1;
                lock_ids[0] = target_locks[4]; // Release a[1]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[2] := 1.
                lock_ids[0] = target_locks[5] = 3 + 2; // Acquire a[2]
                lockManager.acquire_locks(lock_ids, 1);
                a[2] = 1;
                lock_ids[0] = target_locks[5]; // Release a[2]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[3] := 1.
                lock_ids[0] = target_locks[6] = 3 + 3; // Acquire a[3]
                lockManager.acquire_locks(lock_ids, 1);
                a[3] = 1;
                lock_ids[0] = target_locks[6]; // Release a[3]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[4] := 0.
                lock_ids[0] = target_locks[7] = 3 + 4; // Acquire a[4]
                lockManager.acquire_locks(lock_ids, 1);
                a[4] = 0;
                lock_ids[0] = target_locks[7]; // Release a[4]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[5] := 2.
                lock_ids[0] = target_locks[8] = 3 + 5; // Acquire a[5]
                lockManager.acquire_locks(lock_ids, 1);
                a[5] = 2;
                lock_ids[0] = target_locks[8]; // Release a[5]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[6] := 2.
                lock_ids[0] = target_locks[9] = 3 + 6; // Acquire a[6]
                lockManager.acquire_locks(lock_ids, 1);
                a[6] = 2;
                lock_ids[0] = target_locks[9]; // Release a[6]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[7] := 2.
                lock_ids[0] = target_locks[10] = 3 + 7; // Acquire a[7]
                lockManager.acquire_locks(lock_ids, 1);
                a[7] = 2;
                lock_ids[0] = target_locks[10]; // Release a[7]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[8] := 2.
                lock_ids[0] = target_locks[11] = 3 + 8; // Acquire a[8]
                lockManager.acquire_locks(lock_ids, 1);
                a[8] = 2;
                lock_ids[0] = target_locks[11]; // Release a[8]
                lockManager.release_locks(lock_ids, 1);

                currentState = GlobalClass_CheckThread.States.running;
                return true;
            }

            // Attempt to fire a transition starting in state running.
            private void exec_running() {
                // [N_DET.START]
                switch(random.nextInt(6)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | running -> success | tmin > y and fmax < y.
                        if(execute_transition_running_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | running -> failure | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
                        if(execute_transition_running_1()) {
                            return;
                        }
                    }
                    case 2 -> {
                        // SLCO transition (p:0, id:2) | running -> failure | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
                        if(execute_transition_running_2()) {
                            return;
                        }
                    }
                    case 3 -> {
                        // SLCO transition (p:0, id:3) | running -> failure | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
                        if(execute_transition_running_3()) {
                            return;
                        }
                    }
                    case 4 -> {
                        // SLCO transition (p:0, id:4) | running -> failure | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
                        if(execute_transition_running_4()) {
                            return;
                        }
                    }
                    case 5 -> {
                        // SLCO transition (p:0, id:5) | running -> failure | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
                        if(execute_transition_running_5()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state success.
            private void exec_success() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | success -> reset | true.
                if(execute_transition_success_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state failure.
            private void exec_failure() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | failure -> reset | true.
                if(execute_transition_failure_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Attempt to fire a transition starting in state reset.
            private void exec_reset() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | reset -> running | true | [true; y := 4; tmin := 0; fmax := 8; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[4] := 0; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
                if(execute_transition_reset_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 60) {
                    switch(currentState) {
                        case running -> exec_running();
                        case success -> exec_success();
                        case failure -> exec_failure();
                        case reset -> exec_reset();
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
            T_Toad.start();
            T_Frog.start();
            T_Check.start();
        }

        // Join all threads.
        public void joinThreads() {
            while (true) {
                try {
                    T_Toad.join();
                    T_Frog.join();
                    T_Check.join();
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
        ToadsAndFrogs model = new ToadsAndFrogs();
        model.startThreads();
        model.joinThreads();
    }
}