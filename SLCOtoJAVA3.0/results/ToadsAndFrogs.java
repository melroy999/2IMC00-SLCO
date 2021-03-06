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
                4,
                0,
                8,
                new int[]{ 1, 1, 1, 1, 0, 2, 2, 2, 2 }
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
        private final Thread T_toad;
        private final Thread T_frog;
        private final Thread T_control;

        // Class variables.
        private volatile int y;
        private volatile int tmin;
        private volatile int fmax;
        private final int[] a;

        GlobalClass(int y, int tmin, int fmax, int[] a) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(12);

            // Instantiate the class variables.
            this.y = y;
            this.tmin = tmin;
            this.fmax = fmax;
            this.a = a;

            // Instantiate the state machine threads and pass on the class' lock manager.
            T_toad = new GlobalClass_toadThread(lockManager);
            T_frog = new GlobalClass_frogThread(lockManager);
            T_control = new GlobalClass_controlThread(lockManager);
        }

        // Define the states fot the state machine toad.
        interface GlobalClass_toadThread_States {
            enum States {
                q
            }
        }

        // Representation of the SLCO state machine toad.
        class GlobalClass_toadThread extends Thread implements GlobalClass_toadThread_States {
            // Current state
            private GlobalClass_toadThread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            GlobalClass_toadThread(LockManager lockManagerInstance) {
                currentState = GlobalClass_toadThread.States.q;
                lockManager = lockManagerInstance;
                lock_ids = new int[11];
                target_locks = new int[13];
                random = new Random();
            }

            // SLCO expression wrapper | y > 0.
            private boolean t_q_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y > 0) {
                    return true;
                }
                lock_ids[0] = target_locks[1] = 1; // Acquire tmin
                lockManager.acquire_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | tmin != y - 1.
            private boolean t_q_0_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 1; // Acquire tmin
                lockManager.acquire_locks(lock_ids, 1);
                return tmin != y - 1;
            }

            // SLCO expression wrapper | y > 0 and tmin != y - 1.
            private boolean t_q_0_s_0_n_2() {
                if(t_q_0_s_0_n_0() && t_q_0_s_0_n_1()) {
                    return true;
                }
                lock_ids[0] = target_locks[2] = 3 + 4; // Acquire a[4]
                lock_ids[1] = target_locks[3] = 3 + 0; // Acquire a[0]
                lock_ids[2] = target_locks[4] = 3 + 2; // Acquire a[2]
                lock_ids[3] = target_locks[5] = 3 + 1; // Acquire a[1]
                lock_ids[4] = target_locks[6] = 3 + 3; // Acquire a[3]
                lock_ids[5] = target_locks[7] = 3 + 5; // Acquire a[5]
                lock_ids[6] = target_locks[8] = 3 + 6; // Acquire a[6]
                lock_ids[7] = target_locks[9] = 3 + 7; // Acquire a[7]
                lock_ids[8] = target_locks[10] = 3 + 8; // Acquire a[8]
                lockManager.acquire_locks(lock_ids, 9);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 1.
            private boolean t_q_0_s_0_n_3() {
                lock_ids[0] = target_locks[2] = 3 + 4; // Acquire a[4]
                lock_ids[1] = target_locks[3] = 3 + 0; // Acquire a[0]
                lock_ids[2] = target_locks[4] = 3 + 2; // Acquire a[2]
                lock_ids[3] = target_locks[5] = 3 + 1; // Acquire a[1]
                lock_ids[4] = target_locks[6] = 3 + 3; // Acquire a[3]
                lock_ids[5] = target_locks[7] = 3 + 5; // Acquire a[5]
                lock_ids[6] = target_locks[8] = 3 + 6; // Acquire a[6]
                lock_ids[7] = target_locks[9] = 3 + 7; // Acquire a[7]
                lock_ids[8] = target_locks[10] = 3 + 8; // Acquire a[8]
                lock_ids[9] = target_locks[11] = 3 + y - 1; // Acquire a[y - 1]
                lock_ids[10] = target_locks[12] = 3 + y; // Acquire a[y]
                lockManager.acquire_locks(lock_ids, 11);
                if(a[y - 1] == 1) {
                    lock_ids[0] = target_locks[1]; // Release tmin
                    lock_ids[1] = target_locks[2]; // Release a[4]
                    lock_ids[2] = target_locks[3]; // Release a[0]
                    lock_ids[3] = target_locks[4]; // Release a[2]
                    lock_ids[4] = target_locks[5]; // Release a[1]
                    lock_ids[5] = target_locks[6]; // Release a[3]
                    lock_ids[6] = target_locks[7]; // Release a[5]
                    lock_ids[7] = target_locks[8]; // Release a[6]
                    lock_ids[8] = target_locks[9]; // Release a[7]
                    lock_ids[9] = target_locks[10]; // Release a[8]
                    lockManager.release_locks(lock_ids, 10);
                    return true;
                }
                lock_ids[0] = target_locks[11]; // Release a[y - 1]
                lock_ids[1] = target_locks[12]; // Release a[y]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | q -> q | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
            private boolean execute_transition_q_0() {
                // SLCO composite | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
                // SLCO expression | y > 0 and tmin != y - 1 and a[y - 1] = 1.
                if(!(t_q_0_s_0_n_2() && t_q_0_s_0_n_3())) {
                    return false;
                }
                // SLCO assignment | a[y] := 1.
                a[y] = 1;
                lock_ids[0] = target_locks[12]; // Release a[y]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y - 1.
                y = y - 1;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[11]; // Release a[y - 1]
                lockManager.release_locks(lock_ids, 2);

                currentState = GlobalClass_toadThread.States.q;
                return true;
            }

            // SLCO transition (p:0, id:1) | q -> q | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
            private boolean execute_transition_q_1() {
                // SLCO composite | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
                // SLCO expression | y > 0 and tmin = y - 1 and a[y - 1] = 1.
                if(!(y > 0 && tmin == y - 1 && a[y - 1] == 1)) {
                    return false;
                }
                // SLCO assignment | a[y] := 1.
                a[y] = 1;
                // SLCO assignment | tmin := y.
                tmin = y;
                lock_ids[0] = target_locks[1]; // Release tmin
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y - 1.
                y = y - 1;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[4]
                lock_ids[2] = target_locks[3]; // Release a[0]
                lock_ids[3] = target_locks[4]; // Release a[2]
                lock_ids[4] = target_locks[5]; // Release a[1]
                lock_ids[5] = target_locks[6]; // Release a[3]
                lock_ids[6] = target_locks[7]; // Release a[5]
                lock_ids[7] = target_locks[8]; // Release a[6]
                lock_ids[8] = target_locks[9]; // Release a[7]
                lock_ids[9] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);

                currentState = GlobalClass_toadThread.States.q;
                return true;
            }

            // SLCO expression wrapper | a[y - 1] = 2.
            private boolean t_q_2_s_0_n_0() {
                if(a[y - 1] == 2) {
                    lock_ids[0] = target_locks[1]; // Release tmin
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:2) | q -> q | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
            private boolean execute_transition_q_2() {
                // SLCO composite | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
                // SLCO expression | y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
                if(!(y > 1 && tmin != y - 2 && a[y - 2] == 1 && t_q_2_s_0_n_0())) {
                    return false;
                }
                // SLCO assignment | a[y] := 1.
                a[y] = 1;
                // SLCO assignment | y := y - 2.
                y = y - 2;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[4]
                lock_ids[2] = target_locks[3]; // Release a[0]
                lock_ids[3] = target_locks[4]; // Release a[2]
                lock_ids[4] = target_locks[5]; // Release a[1]
                lock_ids[5] = target_locks[6]; // Release a[3]
                lock_ids[6] = target_locks[7]; // Release a[5]
                lock_ids[7] = target_locks[8]; // Release a[6]
                lock_ids[8] = target_locks[9]; // Release a[7]
                lock_ids[9] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);

                currentState = GlobalClass_toadThread.States.q;
                return true;
            }

            // SLCO expression wrapper | y > 1.
            private boolean t_q_3_s_0_n_0() {
                if(y > 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lock_ids[2] = target_locks[2]; // Release a[4]
                lock_ids[3] = target_locks[3]; // Release a[0]
                lock_ids[4] = target_locks[4]; // Release a[2]
                lock_ids[5] = target_locks[5]; // Release a[1]
                lock_ids[6] = target_locks[6]; // Release a[3]
                lock_ids[7] = target_locks[7]; // Release a[5]
                lock_ids[8] = target_locks[8]; // Release a[6]
                lock_ids[9] = target_locks[9]; // Release a[7]
                lock_ids[10] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 11);
                return false;
            }

            // SLCO expression wrapper | tmin = y - 2.
            private boolean t_q_3_s_0_n_1() {
                if(tmin == y - 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lock_ids[2] = target_locks[2]; // Release a[4]
                lock_ids[3] = target_locks[3]; // Release a[0]
                lock_ids[4] = target_locks[4]; // Release a[2]
                lock_ids[5] = target_locks[5]; // Release a[1]
                lock_ids[6] = target_locks[6]; // Release a[3]
                lock_ids[7] = target_locks[7]; // Release a[5]
                lock_ids[8] = target_locks[8]; // Release a[6]
                lock_ids[9] = target_locks[9]; // Release a[7]
                lock_ids[10] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 11);
                return false;
            }

            // SLCO expression wrapper | a[y - 2] = 1.
            private boolean t_q_3_s_0_n_2() {
                if(a[y - 2] == 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lock_ids[2] = target_locks[2]; // Release a[4]
                lock_ids[3] = target_locks[3]; // Release a[0]
                lock_ids[4] = target_locks[4]; // Release a[2]
                lock_ids[5] = target_locks[5]; // Release a[1]
                lock_ids[6] = target_locks[6]; // Release a[3]
                lock_ids[7] = target_locks[7]; // Release a[5]
                lock_ids[8] = target_locks[8]; // Release a[6]
                lock_ids[9] = target_locks[9]; // Release a[7]
                lock_ids[10] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 11);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 2.
            private boolean t_q_3_s_0_n_3() {
                if(a[y - 1] == 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lock_ids[2] = target_locks[2]; // Release a[4]
                lock_ids[3] = target_locks[3]; // Release a[0]
                lock_ids[4] = target_locks[4]; // Release a[2]
                lock_ids[5] = target_locks[5]; // Release a[1]
                lock_ids[6] = target_locks[6]; // Release a[3]
                lock_ids[7] = target_locks[7]; // Release a[5]
                lock_ids[8] = target_locks[8]; // Release a[6]
                lock_ids[9] = target_locks[9]; // Release a[7]
                lock_ids[10] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 11);
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
                // SLCO assignment | tmin := y.
                tmin = y;
                lock_ids[0] = target_locks[1]; // Release tmin
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y - 2.
                y = y - 2;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[4]
                lock_ids[2] = target_locks[3]; // Release a[0]
                lock_ids[3] = target_locks[4]; // Release a[2]
                lock_ids[4] = target_locks[5]; // Release a[1]
                lock_ids[5] = target_locks[6]; // Release a[3]
                lock_ids[6] = target_locks[7]; // Release a[5]
                lock_ids[7] = target_locks[8]; // Release a[6]
                lock_ids[8] = target_locks[9]; // Release a[7]
                lock_ids[9] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);

                currentState = GlobalClass_toadThread.States.q;
                return true;
            }

            // Attempt to fire a transition starting in state q.
            private void exec_q() {
                // [SEQ.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | q -> q | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
                if(execute_transition_q_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | q -> q | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
                if(execute_transition_q_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | q -> q | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
                if(execute_transition_q_2()) {
                    return;
                }
                // SLCO transition (p:0, id:3) | q -> q | [y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; tmin := y; y := y - 2; a[y] := 0].
                if(execute_transition_q_3()) {
                    return;
                }
                // [DET.END]
                // [SEQ.END]
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

        // Define the states fot the state machine frog.
        interface GlobalClass_frogThread_States {
            enum States {
                q
            }
        }

        // Representation of the SLCO state machine frog.
        class GlobalClass_frogThread extends Thread implements GlobalClass_frogThread_States {
            // Current state
            private GlobalClass_frogThread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            GlobalClass_frogThread(LockManager lockManagerInstance) {
                currentState = GlobalClass_frogThread.States.q;
                lockManager = lockManagerInstance;
                lock_ids = new int[11];
                target_locks = new int[13];
                random = new Random();
            }

            // SLCO expression wrapper | y < 8.
            private boolean t_q_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y < 8) {
                    return true;
                }
                lock_ids[0] = target_locks[1] = 2; // Acquire fmax
                lockManager.acquire_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | fmax != y + 1.
            private boolean t_q_0_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 2; // Acquire fmax
                lockManager.acquire_locks(lock_ids, 1);
                return fmax != y + 1;
            }

            // SLCO expression wrapper | y < 8 and fmax != y + 1.
            private boolean t_q_0_s_0_n_2() {
                if(t_q_0_s_0_n_0() && t_q_0_s_0_n_1()) {
                    return true;
                }
                lock_ids[0] = target_locks[2] = 3 + 4; // Acquire a[4]
                lock_ids[1] = target_locks[3] = 3 + 0; // Acquire a[0]
                lock_ids[2] = target_locks[4] = 3 + 2; // Acquire a[2]
                lock_ids[3] = target_locks[5] = 3 + 1; // Acquire a[1]
                lock_ids[4] = target_locks[6] = 3 + 3; // Acquire a[3]
                lock_ids[5] = target_locks[7] = 3 + 5; // Acquire a[5]
                lock_ids[6] = target_locks[8] = 3 + 6; // Acquire a[6]
                lock_ids[7] = target_locks[9] = 3 + 7; // Acquire a[7]
                lock_ids[8] = target_locks[10] = 3 + 8; // Acquire a[8]
                lockManager.acquire_locks(lock_ids, 9);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 2.
            private boolean t_q_0_s_0_n_3() {
                lock_ids[0] = target_locks[2] = 3 + 4; // Acquire a[4]
                lock_ids[1] = target_locks[3] = 3 + 0; // Acquire a[0]
                lock_ids[2] = target_locks[4] = 3 + 2; // Acquire a[2]
                lock_ids[3] = target_locks[5] = 3 + 1; // Acquire a[1]
                lock_ids[4] = target_locks[6] = 3 + 3; // Acquire a[3]
                lock_ids[5] = target_locks[7] = 3 + 5; // Acquire a[5]
                lock_ids[6] = target_locks[8] = 3 + 6; // Acquire a[6]
                lock_ids[7] = target_locks[9] = 3 + 7; // Acquire a[7]
                lock_ids[8] = target_locks[10] = 3 + 8; // Acquire a[8]
                lock_ids[9] = target_locks[11] = 3 + y + 1; // Acquire a[y + 1]
                lock_ids[10] = target_locks[12] = 3 + y; // Acquire a[y]
                lockManager.acquire_locks(lock_ids, 11);
                if(a[y + 1] == 2) {
                    lock_ids[0] = target_locks[1]; // Release fmax
                    lock_ids[1] = target_locks[2]; // Release a[4]
                    lock_ids[2] = target_locks[3]; // Release a[0]
                    lock_ids[3] = target_locks[4]; // Release a[2]
                    lock_ids[4] = target_locks[5]; // Release a[1]
                    lock_ids[5] = target_locks[6]; // Release a[3]
                    lock_ids[6] = target_locks[7]; // Release a[5]
                    lock_ids[7] = target_locks[8]; // Release a[6]
                    lock_ids[8] = target_locks[9]; // Release a[7]
                    lock_ids[9] = target_locks[10]; // Release a[8]
                    lockManager.release_locks(lock_ids, 10);
                    return true;
                }
                lock_ids[0] = target_locks[11]; // Release a[y + 1]
                lock_ids[1] = target_locks[12]; // Release a[y]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | q -> q | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
            private boolean execute_transition_q_0() {
                // SLCO composite | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
                // SLCO expression | y < 8 and fmax != y + 1 and a[y + 1] = 2.
                if(!(t_q_0_s_0_n_2() && t_q_0_s_0_n_3())) {
                    return false;
                }
                // SLCO assignment | a[y] := 2.
                a[y] = 2;
                lock_ids[0] = target_locks[12]; // Release a[y]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y + 1.
                y = y + 1;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[11]; // Release a[y + 1]
                lockManager.release_locks(lock_ids, 2);

                currentState = GlobalClass_frogThread.States.q;
                return true;
            }

            // SLCO transition (p:0, id:1) | q -> q | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
            private boolean execute_transition_q_1() {
                // SLCO composite | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
                // SLCO expression | y < 8 and fmax = y + 1 and a[y + 1] = 2.
                if(!(y < 8 && fmax == y + 1 && a[y + 1] == 2)) {
                    return false;
                }
                // SLCO assignment | a[y] := 2.
                a[y] = 2;
                // SLCO assignment | fmax := y.
                fmax = y;
                lock_ids[0] = target_locks[1]; // Release fmax
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y + 1.
                y = y + 1;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[4]
                lock_ids[2] = target_locks[3]; // Release a[0]
                lock_ids[3] = target_locks[4]; // Release a[2]
                lock_ids[4] = target_locks[5]; // Release a[1]
                lock_ids[5] = target_locks[6]; // Release a[3]
                lock_ids[6] = target_locks[7]; // Release a[5]
                lock_ids[7] = target_locks[8]; // Release a[6]
                lock_ids[8] = target_locks[9]; // Release a[7]
                lock_ids[9] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);

                currentState = GlobalClass_frogThread.States.q;
                return true;
            }

            // SLCO expression wrapper | a[y + 2] = 2.
            private boolean t_q_2_s_0_n_0() {
                if(a[y + 2] == 2) {
                    lock_ids[0] = target_locks[1]; // Release fmax
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:2) | q -> q | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
            private boolean execute_transition_q_2() {
                // SLCO composite | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
                // SLCO expression | y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
                if(!(y < 7 && fmax != y + 2 && a[y + 1] == 1 && t_q_2_s_0_n_0())) {
                    return false;
                }
                // SLCO assignment | a[y] := 2.
                a[y] = 2;
                // SLCO assignment | y := y + 2.
                y = y + 2;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[4]
                lock_ids[2] = target_locks[3]; // Release a[0]
                lock_ids[3] = target_locks[4]; // Release a[2]
                lock_ids[4] = target_locks[5]; // Release a[1]
                lock_ids[5] = target_locks[6]; // Release a[3]
                lock_ids[6] = target_locks[7]; // Release a[5]
                lock_ids[7] = target_locks[8]; // Release a[6]
                lock_ids[8] = target_locks[9]; // Release a[7]
                lock_ids[9] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);

                currentState = GlobalClass_frogThread.States.q;
                return true;
            }

            // SLCO expression wrapper | y < 7.
            private boolean t_q_3_s_0_n_0() {
                if(y < 7) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lock_ids[2] = target_locks[2]; // Release a[4]
                lock_ids[3] = target_locks[3]; // Release a[0]
                lock_ids[4] = target_locks[4]; // Release a[2]
                lock_ids[5] = target_locks[5]; // Release a[1]
                lock_ids[6] = target_locks[6]; // Release a[3]
                lock_ids[7] = target_locks[7]; // Release a[5]
                lock_ids[8] = target_locks[8]; // Release a[6]
                lock_ids[9] = target_locks[9]; // Release a[7]
                lock_ids[10] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 11);
                return false;
            }

            // SLCO expression wrapper | fmax = y + 2.
            private boolean t_q_3_s_0_n_1() {
                if(fmax == y + 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lock_ids[2] = target_locks[2]; // Release a[4]
                lock_ids[3] = target_locks[3]; // Release a[0]
                lock_ids[4] = target_locks[4]; // Release a[2]
                lock_ids[5] = target_locks[5]; // Release a[1]
                lock_ids[6] = target_locks[6]; // Release a[3]
                lock_ids[7] = target_locks[7]; // Release a[5]
                lock_ids[8] = target_locks[8]; // Release a[6]
                lock_ids[9] = target_locks[9]; // Release a[7]
                lock_ids[10] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 11);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 1.
            private boolean t_q_3_s_0_n_2() {
                if(a[y + 1] == 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lock_ids[2] = target_locks[2]; // Release a[4]
                lock_ids[3] = target_locks[3]; // Release a[0]
                lock_ids[4] = target_locks[4]; // Release a[2]
                lock_ids[5] = target_locks[5]; // Release a[1]
                lock_ids[6] = target_locks[6]; // Release a[3]
                lock_ids[7] = target_locks[7]; // Release a[5]
                lock_ids[8] = target_locks[8]; // Release a[6]
                lock_ids[9] = target_locks[9]; // Release a[7]
                lock_ids[10] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 11);
                return false;
            }

            // SLCO expression wrapper | a[y + 2] = 2.
            private boolean t_q_3_s_0_n_3() {
                if(a[y + 2] == 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release fmax
                lock_ids[2] = target_locks[2]; // Release a[4]
                lock_ids[3] = target_locks[3]; // Release a[0]
                lock_ids[4] = target_locks[4]; // Release a[2]
                lock_ids[5] = target_locks[5]; // Release a[1]
                lock_ids[6] = target_locks[6]; // Release a[3]
                lock_ids[7] = target_locks[7]; // Release a[5]
                lock_ids[8] = target_locks[8]; // Release a[6]
                lock_ids[9] = target_locks[9]; // Release a[7]
                lock_ids[10] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 11);
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
                // SLCO assignment | fmax := y.
                fmax = y;
                lock_ids[0] = target_locks[1]; // Release fmax
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | y := y + 2.
                y = y + 2;
                // SLCO assignment | a[y] := 0.
                a[y] = 0;
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release a[4]
                lock_ids[2] = target_locks[3]; // Release a[0]
                lock_ids[3] = target_locks[4]; // Release a[2]
                lock_ids[4] = target_locks[5]; // Release a[1]
                lock_ids[5] = target_locks[6]; // Release a[3]
                lock_ids[6] = target_locks[7]; // Release a[5]
                lock_ids[7] = target_locks[8]; // Release a[6]
                lock_ids[8] = target_locks[9]; // Release a[7]
                lock_ids[9] = target_locks[10]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);

                currentState = GlobalClass_frogThread.States.q;
                return true;
            }

            // Attempt to fire a transition starting in state q.
            private void exec_q() {
                // [SEQ.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | q -> q | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
                if(execute_transition_q_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | q -> q | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
                if(execute_transition_q_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | q -> q | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
                if(execute_transition_q_2()) {
                    return;
                }
                // SLCO transition (p:0, id:3) | q -> q | [y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; fmax := y; y := y + 2; a[y] := 0].
                if(execute_transition_q_3()) {
                    return;
                }
                // [DET.END]
                // [SEQ.END]
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

        // Define the states fot the state machine control.
        interface GlobalClass_controlThread_States {
            enum States {
                running, 
                done, 
                success, 
                failure, 
                reset
            }
        }

        // Representation of the SLCO state machine control.
        class GlobalClass_controlThread extends Thread implements GlobalClass_controlThread_States {
            // Current state
            private GlobalClass_controlThread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            GlobalClass_controlThread(LockManager lockManagerInstance) {
                currentState = GlobalClass_controlThread.States.running;
                lockManager = lockManagerInstance;
                lock_ids = new int[11];
                target_locks = new int[12];
                random = new Random();
            }

            // SLCO expression wrapper | y = 0.
            private boolean t_running_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                if(y == 0) {
                    return true;
                }
                lock_ids[0] = target_locks[1] = 3 + 4; // Acquire a[4]
                lock_ids[1] = target_locks[2] = 3 + 0; // Acquire a[0]
                lock_ids[2] = target_locks[3] = 3 + 2; // Acquire a[2]
                lock_ids[3] = target_locks[4] = 3 + 1; // Acquire a[1]
                lock_ids[4] = target_locks[5] = 3 + 3; // Acquire a[3]
                lock_ids[5] = target_locks[6] = 3 + 5; // Acquire a[5]
                lock_ids[6] = target_locks[7] = 3 + 6; // Acquire a[6]
                lock_ids[7] = target_locks[8] = 3 + 7; // Acquire a[7]
                lock_ids[8] = target_locks[9] = 3 + 8; // Acquire a[8]
                lockManager.acquire_locks(lock_ids, 9);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 1.
            private boolean t_running_0_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 3 + 4; // Acquire a[4]
                lock_ids[1] = target_locks[2] = 3 + 0; // Acquire a[0]
                lock_ids[2] = target_locks[3] = 3 + 2; // Acquire a[2]
                lock_ids[3] = target_locks[4] = 3 + 1; // Acquire a[1]
                lock_ids[4] = target_locks[5] = 3 + 3; // Acquire a[3]
                lock_ids[5] = target_locks[6] = 3 + 5; // Acquire a[5]
                lock_ids[6] = target_locks[7] = 3 + 6; // Acquire a[6]
                lock_ids[7] = target_locks[8] = 3 + 7; // Acquire a[7]
                lock_ids[8] = target_locks[9] = 3 + 8; // Acquire a[8]
                lock_ids[9] = target_locks[10] = 3 + y + 1; // Acquire a[y + 1]
                lock_ids[10] = target_locks[11] = 3 + y + 2; // Acquire a[y + 2]
                lockManager.acquire_locks(lock_ids, 11);
                if(a[y + 1] == 1) {
                    lock_ids[0] = target_locks[10]; // Release a[y + 1]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[10]; // Release a[y + 1]
                lock_ids[1] = target_locks[11]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | a[y + 2] = 1.
            private boolean t_running_0_s_0_n_2() {
                if(a[y + 2] == 1) {
                    lock_ids[0] = target_locks[0]; // Release y
                    lock_ids[1] = target_locks[1]; // Release a[4]
                    lock_ids[2] = target_locks[2]; // Release a[0]
                    lock_ids[3] = target_locks[3]; // Release a[2]
                    lock_ids[4] = target_locks[4]; // Release a[1]
                    lock_ids[5] = target_locks[5]; // Release a[3]
                    lock_ids[6] = target_locks[6]; // Release a[5]
                    lock_ids[7] = target_locks[7]; // Release a[6]
                    lock_ids[8] = target_locks[8]; // Release a[7]
                    lock_ids[9] = target_locks[9]; // Release a[8]
                    lock_ids[10] = target_locks[11]; // Release a[y + 2]
                    lockManager.release_locks(lock_ids, 11);
                    return true;
                }
                lock_ids[0] = target_locks[11]; // Release a[y + 2]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:0) | running -> done | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
            private boolean execute_transition_running_0() {
                // SLCO expression | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
                if(!(t_running_0_s_0_n_0() && t_running_0_s_0_n_1() && t_running_0_s_0_n_2())) {
                    return false;
                }

                currentState = GlobalClass_controlThread.States.done;
                return true;
            }

            // SLCO expression wrapper | a[y + 2] = 1.
            private boolean t_running_1_s_0_n_0() {
                if(a[y + 2] == 1) {
                    lock_ids[0] = target_locks[0]; // Release y
                    lock_ids[1] = target_locks[1]; // Release a[4]
                    lock_ids[2] = target_locks[2]; // Release a[0]
                    lock_ids[3] = target_locks[3]; // Release a[2]
                    lock_ids[4] = target_locks[4]; // Release a[1]
                    lock_ids[5] = target_locks[5]; // Release a[3]
                    lock_ids[6] = target_locks[6]; // Release a[5]
                    lock_ids[7] = target_locks[7]; // Release a[6]
                    lock_ids[8] = target_locks[8]; // Release a[7]
                    lock_ids[9] = target_locks[9]; // Release a[8]
                    lockManager.release_locks(lock_ids, 10);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:1) | running -> done | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
            private boolean execute_transition_running_1() {
                // SLCO expression | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
                if(!(y == 1 && a[y - 1] == 2 && a[y + 1] == 1 && t_running_1_s_0_n_0())) {
                    return false;
                }

                currentState = GlobalClass_controlThread.States.done;
                return true;
            }

            // SLCO expression wrapper | a[y + 1] = 1.
            private boolean t_running_2_s_0_n_0() {
                if(a[y + 1] == 1) {
                    lock_ids[0] = target_locks[0]; // Release y
                    lock_ids[1] = target_locks[1]; // Release a[4]
                    lock_ids[2] = target_locks[2]; // Release a[0]
                    lock_ids[3] = target_locks[3]; // Release a[2]
                    lock_ids[4] = target_locks[4]; // Release a[1]
                    lock_ids[5] = target_locks[5]; // Release a[3]
                    lock_ids[6] = target_locks[6]; // Release a[5]
                    lock_ids[7] = target_locks[7]; // Release a[6]
                    lock_ids[8] = target_locks[8]; // Release a[7]
                    lock_ids[9] = target_locks[9]; // Release a[8]
                    lockManager.release_locks(lock_ids, 10);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:2) | running -> done | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
            private boolean execute_transition_running_2() {
                // SLCO expression | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
                if(!(y == 7 && a[y - 2] == 2 && a[y - 1] == 2 && t_running_2_s_0_n_0())) {
                    return false;
                }

                currentState = GlobalClass_controlThread.States.done;
                return true;
            }

            // SLCO expression wrapper | a[y - 1] = 2.
            private boolean t_running_3_s_0_n_0() {
                if(a[y - 1] == 2) {
                    lock_ids[0] = target_locks[0]; // Release y
                    lock_ids[1] = target_locks[1]; // Release a[4]
                    lock_ids[2] = target_locks[2]; // Release a[0]
                    lock_ids[3] = target_locks[3]; // Release a[2]
                    lock_ids[4] = target_locks[4]; // Release a[1]
                    lock_ids[5] = target_locks[5]; // Release a[3]
                    lock_ids[6] = target_locks[6]; // Release a[5]
                    lock_ids[7] = target_locks[7]; // Release a[6]
                    lock_ids[8] = target_locks[8]; // Release a[7]
                    lock_ids[9] = target_locks[9]; // Release a[8]
                    lockManager.release_locks(lock_ids, 10);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:3) | running -> done | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
            private boolean execute_transition_running_3() {
                // SLCO expression | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
                if(!(y == 8 && a[y - 2] == 2 && t_running_3_s_0_n_0())) {
                    return false;
                }

                currentState = GlobalClass_controlThread.States.done;
                return true;
            }

            // SLCO expression wrapper | y > 1.
            private boolean t_running_4_s_0_n_0() {
                if(y > 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release a[4]
                lock_ids[2] = target_locks[2]; // Release a[0]
                lock_ids[3] = target_locks[3]; // Release a[2]
                lock_ids[4] = target_locks[4]; // Release a[1]
                lock_ids[5] = target_locks[5]; // Release a[3]
                lock_ids[6] = target_locks[6]; // Release a[5]
                lock_ids[7] = target_locks[7]; // Release a[6]
                lock_ids[8] = target_locks[8]; // Release a[7]
                lock_ids[9] = target_locks[9]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);
                return false;
            }

            // SLCO expression wrapper | y < 7.
            private boolean t_running_4_s_0_n_1() {
                if(y < 7) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release a[4]
                lock_ids[2] = target_locks[2]; // Release a[0]
                lock_ids[3] = target_locks[3]; // Release a[2]
                lock_ids[4] = target_locks[4]; // Release a[1]
                lock_ids[5] = target_locks[5]; // Release a[3]
                lock_ids[6] = target_locks[6]; // Release a[5]
                lock_ids[7] = target_locks[7]; // Release a[6]
                lock_ids[8] = target_locks[8]; // Release a[7]
                lock_ids[9] = target_locks[9]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);
                return false;
            }

            // SLCO expression wrapper | a[y - 2] = 2.
            private boolean t_running_4_s_0_n_2() {
                if(a[y - 2] == 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release a[4]
                lock_ids[2] = target_locks[2]; // Release a[0]
                lock_ids[3] = target_locks[3]; // Release a[2]
                lock_ids[4] = target_locks[4]; // Release a[1]
                lock_ids[5] = target_locks[5]; // Release a[3]
                lock_ids[6] = target_locks[6]; // Release a[5]
                lock_ids[7] = target_locks[7]; // Release a[6]
                lock_ids[8] = target_locks[8]; // Release a[7]
                lock_ids[9] = target_locks[9]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);
                return false;
            }

            // SLCO expression wrapper | a[y - 1] = 2.
            private boolean t_running_4_s_0_n_3() {
                if(a[y - 1] == 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release a[4]
                lock_ids[2] = target_locks[2]; // Release a[0]
                lock_ids[3] = target_locks[3]; // Release a[2]
                lock_ids[4] = target_locks[4]; // Release a[1]
                lock_ids[5] = target_locks[5]; // Release a[3]
                lock_ids[6] = target_locks[6]; // Release a[5]
                lock_ids[7] = target_locks[7]; // Release a[6]
                lock_ids[8] = target_locks[8]; // Release a[7]
                lock_ids[9] = target_locks[9]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);
                return false;
            }

            // SLCO expression wrapper | a[y + 1] = 1.
            private boolean t_running_4_s_0_n_4() {
                if(a[y + 1] == 1) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release a[4]
                lock_ids[2] = target_locks[2]; // Release a[0]
                lock_ids[3] = target_locks[3]; // Release a[2]
                lock_ids[4] = target_locks[4]; // Release a[1]
                lock_ids[5] = target_locks[5]; // Release a[3]
                lock_ids[6] = target_locks[6]; // Release a[5]
                lock_ids[7] = target_locks[7]; // Release a[6]
                lock_ids[8] = target_locks[8]; // Release a[7]
                lock_ids[9] = target_locks[9]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);
                return false;
            }

            // SLCO expression wrapper | a[y + 2] = 1.
            private boolean t_running_4_s_0_n_5() {
                if(a[y + 2] == 1) {
                    lock_ids[0] = target_locks[0]; // Release y
                    lock_ids[1] = target_locks[1]; // Release a[4]
                    lock_ids[2] = target_locks[2]; // Release a[0]
                    lock_ids[3] = target_locks[3]; // Release a[2]
                    lock_ids[4] = target_locks[4]; // Release a[1]
                    lock_ids[5] = target_locks[5]; // Release a[3]
                    lock_ids[6] = target_locks[6]; // Release a[5]
                    lock_ids[7] = target_locks[7]; // Release a[6]
                    lock_ids[8] = target_locks[8]; // Release a[7]
                    lock_ids[9] = target_locks[9]; // Release a[8]
                    lockManager.release_locks(lock_ids, 10);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release a[4]
                lock_ids[2] = target_locks[2]; // Release a[0]
                lock_ids[3] = target_locks[3]; // Release a[2]
                lock_ids[4] = target_locks[4]; // Release a[1]
                lock_ids[5] = target_locks[5]; // Release a[3]
                lock_ids[6] = target_locks[6]; // Release a[5]
                lock_ids[7] = target_locks[7]; // Release a[6]
                lock_ids[8] = target_locks[8]; // Release a[7]
                lock_ids[9] = target_locks[9]; // Release a[8]
                lockManager.release_locks(lock_ids, 10);
                return false;
            }

            // SLCO transition (p:0, id:4) | running -> done | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
            private boolean execute_transition_running_4() {
                // SLCO expression | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
                if(!(t_running_4_s_0_n_0() && t_running_4_s_0_n_1() && t_running_4_s_0_n_2() && t_running_4_s_0_n_3() && t_running_4_s_0_n_4() && t_running_4_s_0_n_5())) {
                    return false;
                }

                currentState = GlobalClass_controlThread.States.done;
                return true;
            }

            // SLCO expression wrapper | tmin > y.
            private boolean t_done_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire y
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[1] = 1; // Acquire tmin
                lockManager.acquire_locks(lock_ids, 1);
                if(tmin > y) {
                    return true;
                }
                lock_ids[0] = target_locks[2] = 2; // Acquire fmax
                lockManager.acquire_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | fmax < y.
            private boolean t_done_0_s_0_n_1() {
                lock_ids[0] = target_locks[2] = 2; // Acquire fmax
                lockManager.acquire_locks(lock_ids, 1);
                if(fmax < y) {
                    lock_ids[0] = target_locks[0]; // Release y
                    lock_ids[1] = target_locks[1]; // Release tmin
                    lock_ids[2] = target_locks[2]; // Release fmax
                    lockManager.release_locks(lock_ids, 3);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:0) | done -> success | tmin > y and fmax < y.
            private boolean execute_transition_done_0() {
                // SLCO expression | tmin > y and fmax < y.
                if(!(t_done_0_s_0_n_0() && t_done_0_s_0_n_1())) {
                    return false;
                }

                currentState = GlobalClass_controlThread.States.success;
                return true;
            }

            // SLCO expression wrapper | tmin > y.
            private boolean t_done_1_s_0_n_0() {
                if(tmin > y) {
                    lock_ids[0] = target_locks[1]; // Release tmin
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[1]; // Release tmin
                lock_ids[2] = target_locks[2]; // Release fmax
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | fmax < y.
            private boolean t_done_1_s_0_n_1() {
                if(fmax < y) {
                    lock_ids[0] = target_locks[0]; // Release y
                    lock_ids[1] = target_locks[2]; // Release fmax
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release y
                lock_ids[1] = target_locks[2]; // Release fmax
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:1) | done -> failure | !(tmin > y and fmax < y).
            private boolean execute_transition_done_1() {
                // SLCO expression | !(tmin > y and fmax < y).
                if(!(!((t_done_1_s_0_n_0() && t_done_1_s_0_n_1())))) {
                    return false;
                }

                currentState = GlobalClass_controlThread.States.failure;
                return true;
            }

            // SLCO transition (p:0, id:0) | success -> reset | true.
            private boolean execute_transition_success_0() {
                // (Superfluous) SLCO expression | true.

                currentState = GlobalClass_controlThread.States.reset;
                return true;
            }

            // SLCO transition (p:0, id:0) | failure -> reset | true.
            private boolean execute_transition_failure_0() {
                // (Superfluous) SLCO expression | true.

                currentState = GlobalClass_controlThread.States.reset;
                return true;
            }

            // SLCO transition (p:0, id:0) | reset -> running | true | [true; y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
            private boolean execute_transition_reset_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2] -> [true; y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
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
                // SLCO assignment | a[4] := 0.
                lock_ids[0] = target_locks[3] = 3 + 2; // Acquire a[2]
                lock_ids[1] = target_locks[4] = 3 + 4; // Acquire a[4]
                lock_ids[2] = target_locks[5] = 3 + 3; // Acquire a[3]
                lock_ids[3] = target_locks[6] = 3 + 0; // Acquire a[0]
                lock_ids[4] = target_locks[7] = 3 + 1; // Acquire a[1]
                lockManager.acquire_locks(lock_ids, 5);
                a[4] = 0;
                lock_ids[0] = target_locks[4]; // Release a[4]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[0] := 1.
                a[0] = 1;
                lock_ids[0] = target_locks[6]; // Release a[0]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[1] := 1.
                a[1] = 1;
                lock_ids[0] = target_locks[7]; // Release a[1]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[2] := 1.
                a[2] = 1;
                lock_ids[0] = target_locks[3]; // Release a[2]
                lockManager.release_locks(lock_ids, 1);
                // SLCO assignment | a[3] := 1.
                a[3] = 1;
                lock_ids[0] = target_locks[5]; // Release a[3]
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

                currentState = GlobalClass_controlThread.States.running;
                return true;
            }

            // Attempt to fire a transition starting in state running.
            private void exec_running() {
                // [SEQ.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | running -> done | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
                if(execute_transition_running_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | running -> done | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
                if(execute_transition_running_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | running -> done | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
                if(execute_transition_running_2()) {
                    return;
                }
                // SLCO transition (p:0, id:3) | running -> done | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
                if(execute_transition_running_3()) {
                    return;
                }
                // SLCO transition (p:0, id:4) | running -> done | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
                if(execute_transition_running_4()) {
                    return;
                }
                // [DET.END]
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state done.
            private void exec_done() {
                // [SEQ.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | done -> success | tmin > y and fmax < y.
                if(execute_transition_done_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | done -> failure | !(tmin > y and fmax < y).
                if(execute_transition_done_1()) {
                    return;
                }
                // [DET.END]
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state success.
            private void exec_success() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | success -> reset | true.
                if(execute_transition_success_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state failure.
            private void exec_failure() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | failure -> reset | true.
                if(execute_transition_failure_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state reset.
            private void exec_reset() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | reset -> running | true | [true; y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
                if(execute_transition_reset_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 60) {
                    switch(currentState) {
                        case running -> exec_running();
                        case done -> exec_done();
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
            T_toad.start();
            T_frog.start();
            T_control.start();
        }

        // Join all threads.
        public void joinThreads() {
            while (true) {
                try {
                    T_toad.join();
                    T_frog.join();
                    T_control.join();
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