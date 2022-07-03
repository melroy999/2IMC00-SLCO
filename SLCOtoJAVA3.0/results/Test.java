package testing;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.time.Duration;
import java.time.Instant;

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
                new int[]{ 0, 0 },
                new boolean[]{ false, false, false },
                0,
                new int[]{ 0, 0 },
                (char) 0
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
        private final int[] x;
        private final int[] y;
        private final boolean[] b;
        private volatile int i;
        private final int[] c;
        private volatile int d;

        P(int[] x, int[] y, boolean[] b, int i, int[] c, int d) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(11);

            // Instantiate the class variables.
            this.x = x;
            this.y = y;
            this.b = b;
            this.i = i;
            this.c = c;
            this.d = d;

            // Instantiate the state machine threads and pass on the class' lock manager.
            T_SM1 = new P_SM1Thread(lockManager);
        }

        // Define the states fot the state machine SM1.
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
                lock_ids = new int[4];
                target_locks = new int[5];
                random = new Random();
            }

            // SLCO expression wrapper | x[0] > 0.
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[2] = 5; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 9 + 0; // Acquire x[0]
                lock_ids[1] = target_locks[1] = 9 + (i + 1); // Acquire x[(i + 1)]
                lockManager.acquire_locks(lock_ids, 2);
                if(x[0] > 0) {
                    lock_ids[0] = target_locks[0]; // Release x[0]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[1]; // Release x[(i + 1)]
                lock_ids[2] = target_locks[2]; // Release i
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SimpleState0 | [x[0] > 0; i := i + 1; x[i] := 1].
            private boolean execute_transition_SMC0_0() {
                // SLCO composite | [x[0] > 0; i := i + 1; x[i] := 1].
                // SLCO expression | x[0] > 0.
                if(!(t_SMC0_0_s_0_n_0())) {
                    return false;
                }
                // SLCO assignment | i := i + 1.
                i = i + 1;
                // SLCO assignment | x[i] := 1.
                x[i] = 1;
                lock_ids[0] = target_locks[1]; // Release x[(i + 1)]
                lock_ids[1] = target_locks[2]; // Release i
                lockManager.release_locks(lock_ids, 2);

                currentState = P_SM1Thread.States.SimpleState0;
                return true;
            }

            // SLCO expression wrapper | x[0] > 0.
            private boolean t_SimpleState0_0_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 5; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 9 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 1);
                if(x[0] > 0) {
                    lock_ids[0] = target_locks[0]; // Release x[0]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[1]; // Release i
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SimpleState0 -> SimpleState1 | [x[0] > 0; i := 2; x[i] := 1; x[1] := 1].
            private boolean execute_transition_SimpleState0_0() {
                // SLCO composite | [x[0] > 0; i := 2; x[i] := 1; x[1] := 1].
                // SLCO expression | x[0] > 0.
                if(!(t_SimpleState0_0_s_0_n_0())) {
                    return false;
                }
                // SLCO assignment | i := 2.
                lock_ids[0] = target_locks[2] = 9 + 1; // Acquire x[1]
                lock_ids[1] = target_locks[3] = 9 + 2; // Acquire x[2]
                lockManager.acquire_locks(lock_ids, 2);
                i = 2;
                // SLCO assignment | x[i] := 1.
                x[i] = 1;
                lock_ids[0] = target_locks[1]; // Release i
                lock_ids[1] = target_locks[3]; // Release x[2]
                lockManager.release_locks(lock_ids, 2);
                // SLCO assignment | x[1] := 1.
                x[1] = 1;
                lock_ids[0] = target_locks[2]; // Release x[1]
                lockManager.release_locks(lock_ids, 1);

                currentState = P_SM1Thread.States.SimpleState1;
                return true;
            }

            // SLCO expression wrapper | x[0] > 0.
            private boolean t_SimpleState1_0_s_0_n_0() {
                lock_ids[0] = target_locks[1] = 5; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 9 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 1);
                if(x[0] > 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[1]; // Release i
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SimpleState1 -> SimpleState2 | [x[0] > 0; i := 0; x[i] := 1; x[1] := 1].
            private boolean execute_transition_SimpleState1_0() {
                // SLCO composite | [x[0] > 0; i := 0; x[i] := 1; x[1] := 1].
                // SLCO expression | x[0] > 0.
                if(!(t_SimpleState1_0_s_0_n_0())) {
                    return false;
                }
                // SLCO assignment | i := 0.
                i = 0;
                // SLCO assignment | x[i] := 1.
                lock_ids[0] = target_locks[2] = 9 + 1; // Acquire x[1]
                lockManager.acquire_locks(lock_ids, 1);
                x[i] = 1;
                lock_ids[0] = target_locks[0]; // Release x[0]
                lock_ids[1] = target_locks[1]; // Release i
                lockManager.release_locks(lock_ids, 2);
                // SLCO assignment | x[1] := 1.
                x[1] = 1;
                lock_ids[0] = target_locks[2]; // Release x[1]
                lockManager.release_locks(lock_ids, 1);

                currentState = P_SM1Thread.States.SimpleState2;
                return true;
            }

            // SLCO expression wrapper | i >= 0.
            private boolean t_SimpleState2_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 5; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 3.
            private boolean t_SimpleState2_0_s_0_n_1() {
                if(i < 3) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | b[i].
            private boolean t_SimpleState2_0_s_0_n_2() {
                lock_ids[0] = target_locks[1] = 6 + i; // Acquire b[i]
                lockManager.acquire_locks(lock_ids, 1);
                if(b[i]) {
                    lock_ids[0] = target_locks[1]; // Release b[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[1]; // Release b[i]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SimpleState2 -> SimpleState3 | [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1].
            private boolean execute_transition_SimpleState2_0() {
                // SLCO composite | [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1].
                // SLCO expression | i >= 0 and i < 3 and b[i].
                if(!(t_SimpleState2_0_s_0_n_0() && t_SimpleState2_0_s_0_n_1() && t_SimpleState2_0_s_0_n_2())) {
                    return false;
                }
                // SLCO assignment | i := 0.
                i = 0;
                // SLCO assignment | x[i] := 1.
                lock_ids[0] = target_locks[2] = 9 + i; // Acquire x[i]
                lock_ids[1] = target_locks[3] = 9 + 1; // Acquire x[1]
                lockManager.acquire_locks(lock_ids, 2);
                x[i] = 1;
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[2]; // Release x[i]
                lockManager.release_locks(lock_ids, 2);
                // SLCO assignment | x[1] := 1.
                x[1] = 1;
                lock_ids[0] = target_locks[3]; // Release x[1]
                lockManager.release_locks(lock_ids, 1);

                currentState = P_SM1Thread.States.SimpleState3;
                return true;
            }

            // SLCO expression wrapper | i >= 0.
            private boolean t_SimpleState3_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 5; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO expression wrapper | i < 2.
            private boolean t_SimpleState3_0_s_0_n_1() {
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:0) | SimpleState3 -> SimpleState4 | [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0].
            private boolean execute_transition_SimpleState3_0() {
                // SLCO composite | [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0].
                // SLCO expression | i >= 0 and i < 2.
                if(!(t_SimpleState3_0_s_0_n_0() && t_SimpleState3_0_s_0_n_1())) {
                    return false;
                }
                // SLCO assignment | i := 0.
                i = 0;
                // SLCO assignment | x[i] := 0.
                lock_ids[0] = target_locks[1] = 9 + i; // Acquire x[i]
                lock_ids[1] = target_locks[2] = 9 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 2);
                x[i] = 0;
                // SLCO assignment | x[x[i]] := 0.
                x[x[i]] = 0;
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[1]; // Release x[i]
                lock_ids[2] = target_locks[2]; // Release x[0]
                lockManager.release_locks(lock_ids, 3);

                currentState = P_SM1Thread.States.SimpleState4;
                return true;
            }

            // SLCO expression wrapper | i >= 0.
            private boolean t_SimpleState4_0_s_3_n_0() {
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release b[1]
                lock_ids[1] = target_locks[3]; // Release b[2]
                lock_ids[2] = target_locks[4]; // Release b[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | i < 2.
            private boolean t_SimpleState4_0_s_3_n_1() {
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release b[1]
                lock_ids[1] = target_locks[3]; // Release b[2]
                lock_ids[2] = target_locks[4]; // Release b[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO expression wrapper | b[i].
            private boolean t_SimpleState4_0_s_3_n_2() {
                if(b[i]) {
                    lock_ids[0] = target_locks[2]; // Release b[1]
                    lock_ids[1] = target_locks[3]; // Release b[2]
                    lock_ids[2] = target_locks[4]; // Release b[0]
                    lockManager.release_locks(lock_ids, 3);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release b[1]
                lock_ids[1] = target_locks[3]; // Release b[2]
                lock_ids[2] = target_locks[4]; // Release b[0]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            // SLCO transition (p:0, id:0) | SimpleState4 -> SimpleState5 | true | [true; i := 0; b[i] := i >= 0 and i < 2 and b[i]].
            private boolean execute_transition_SimpleState4_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [i := 0; b[i] := i >= 0 and i < 2 and b[i]] -> [true; i := 0; b[i] := i >= 0 and i < 2 and b[i]].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | i := 0.
                lock_ids[0] = target_locks[0] = 5; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                i = 0;
                // SLCO assignment | b[i] := i >= 0 and i < 2 and b[i].
                lock_ids[0] = target_locks[1] = 6 + i; // Acquire b[i]
                lock_ids[1] = target_locks[2] = 6 + 1; // Acquire b[1]
                lock_ids[2] = target_locks[3] = 6 + 2; // Acquire b[2]
                lock_ids[3] = target_locks[4] = 6 + 0; // Acquire b[0]
                lockManager.acquire_locks(lock_ids, 4);
                b[i] = t_SimpleState4_0_s_3_n_0() && t_SimpleState4_0_s_3_n_1() && t_SimpleState4_0_s_3_n_2();
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[1]; // Release b[i]
                lockManager.release_locks(lock_ids, 2);

                currentState = P_SM1Thread.States.SimpleState5;
                return true;
            }

            // SLCO expression wrapper | i >= 0.
            private boolean t_SimpleState5_0_s_4_n_0() {
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release b[1]
                lock_ids[1] = target_locks[3]; // Release b[2]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | i < 2.
            private boolean t_SimpleState5_0_s_4_n_1() {
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release b[1]
                lock_ids[1] = target_locks[3]; // Release b[2]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO expression wrapper | b[i].
            private boolean t_SimpleState5_0_s_4_n_2() {
                if(b[i]) {
                    lock_ids[0] = target_locks[2]; // Release b[1]
                    lock_ids[1] = target_locks[3]; // Release b[2]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release b[1]
                lock_ids[1] = target_locks[3]; // Release b[2]
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (p:0, id:0) | SimpleState5 -> SimpleState6 | true | [true; b[0] := true; i := 0; b[i] := i >= 0 and i < 2 and b[i]].
            private boolean execute_transition_SimpleState5_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO composite | [b[0] := true; i := 0; b[i] := i >= 0 and i < 2 and b[i]] -> [true; b[0] := true; i := 0; b[i] := i >= 0 and i < 2 and b[i]].
                // (Superfluous) SLCO expression | true.
                // SLCO assignment | b[0] := true.
                lock_ids[0] = target_locks[1] = 5; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[0] = 6 + 0; // Acquire b[0]
                lock_ids[1] = target_locks[2] = 6 + 1; // Acquire b[1]
                lock_ids[2] = target_locks[3] = 6 + 2; // Acquire b[2]
                lockManager.acquire_locks(lock_ids, 3);
                b[0] = true;
                // SLCO assignment | i := 0.
                i = 0;
                // SLCO assignment | b[i] := i >= 0 and i < 2 and b[i].
                b[i] = t_SimpleState5_0_s_4_n_0() && t_SimpleState5_0_s_4_n_1() && t_SimpleState5_0_s_4_n_2();
                lock_ids[0] = target_locks[0]; // Release b[0]
                lock_ids[1] = target_locks[1]; // Release i
                lockManager.release_locks(lock_ids, 2);

                currentState = P_SM1Thread.States.SimpleState6;
                return true;
            }

            // SLCO transition (p:0, id:0) | SimpleState6 -> SimpleState7 | true | d := 0.
            private boolean execute_transition_SimpleState6_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | d := 0.
                lock_ids[0] = target_locks[0] = 2; // Acquire d
                lockManager.acquire_locks(lock_ids, 1);
                d = (0) & 0xff;
                lock_ids[0] = target_locks[0]; // Release d
                lockManager.release_locks(lock_ids, 1);

                currentState = P_SM1Thread.States.SimpleState7;
                return true;
            }

            // SLCO transition (p:0, id:0) | SimpleState7 -> SMC0 | true | c[0] := 0.
            private boolean execute_transition_SimpleState7_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | c[0] := 0.
                lock_ids[0] = target_locks[0] = 0 + 0; // Acquire c[0]
                lockManager.acquire_locks(lock_ids, 1);
                c[0] = (0) & 0xff;
                lock_ids[0] = target_locks[0]; // Release c[0]
                lockManager.release_locks(lock_ids, 1);

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // Attempt to fire a transition starting in state SMC0.
            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC0 -> SimpleState0 | [x[0] > 0; i := i + 1; x[i] := 1].
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SMC1.
            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            // Attempt to fire a transition starting in state SimpleState0.
            private void exec_SimpleState0() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SimpleState0 -> SimpleState1 | [x[0] > 0; i := 2; x[i] := 1; x[1] := 1].
                if(execute_transition_SimpleState0_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SimpleState1.
            private void exec_SimpleState1() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SimpleState1 -> SimpleState2 | [x[0] > 0; i := 0; x[i] := 1; x[1] := 1].
                if(execute_transition_SimpleState1_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SimpleState2.
            private void exec_SimpleState2() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SimpleState2 -> SimpleState3 | [i >= 0 and i < 3 and b[i]; i := 0; x[i] := 1; x[1] := 1].
                if(execute_transition_SimpleState2_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SimpleState3.
            private void exec_SimpleState3() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SimpleState3 -> SimpleState4 | [i >= 0 and i < 2; i := 0; x[i] := 0; x[x[i]] := 0].
                if(execute_transition_SimpleState3_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SimpleState4.
            private void exec_SimpleState4() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SimpleState4 -> SimpleState5 | true | [true; i := 0; b[i] := i >= 0 and i < 2 and b[i]].
                if(execute_transition_SimpleState4_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SimpleState5.
            private void exec_SimpleState5() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SimpleState5 -> SimpleState6 | true | [true; b[0] := true; i := 0; b[i] := i >= 0 and i < 2 and b[i]].
                if(execute_transition_SimpleState5_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SimpleState6.
            private void exec_SimpleState6() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SimpleState6 -> SimpleState7 | true | d := 0.
                if(execute_transition_SimpleState6_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Attempt to fire a transition starting in state SimpleState7.
            private void exec_SimpleState7() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SimpleState7 -> SMC0 | true | c[0] := 0.
                if(execute_transition_SimpleState7_0()) {
                    return;
                }
                // [SEQ.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 60) {
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
        Test model = new Test();
        model.startThreads();
        model.joinThreads();
    }
}