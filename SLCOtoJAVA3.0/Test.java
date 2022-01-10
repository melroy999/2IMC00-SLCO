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
                NewState3, 
                NewState4, 
                NewState5, 
                NewState6, 
                NewState7, 
                NewState8, 
                SimpleState0
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
                lock_ids = new int[3];
                target_locks = new int[4];
                random = new Random();
            }

            private boolean t_NewState0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                if(i > 5) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState0 -> SimpleState0
            private boolean execute_transition_NewState0_0() {
                // SLCO composite [i > 5; i := y[i]; x[y[i]] := 1] -> [i > 5; i := y[i]; x[y[i]] := 1] 
                if(!(t_NewState0_0_s_0_n_0())) {
                    return false;
                }
                lock_ids[0] = target_locks[3] = 4 + 0; // Acquire y[0]
                lock_ids[1] = target_locks[2] = 4 + 1; // Acquire y[1]
                lock_ids[2] = target_locks[1] = 4 + i; // Acquire y[i]
                lockManager.acquire_locks(lock_ids, 3);
                lock_ids[0] = target_locks[4] = 4 + y[i]; // Acquire y[y[i]]
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[3]; // Release y[0]
                lock_ids[1] = target_locks[2]; // Release y[1]
                lockManager.release_locks(lock_ids, 2);
                i = y[i];
                lock_ids[0] = target_locks[1]; // Release y[i]
                lockManager.release_locks(lock_ids, 1);
                lock_ids[0] = target_locks[5] = 6 + y[i]; // Acquire x[y[i]]
                lockManager.acquire_locks(lock_ids, 1);
                x[y[i]] = 1;
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[5]; // Release x[y[i]]
                lock_ids[2] = target_locks[4]; // Release y[y[i]]
                lockManager.release_locks(lock_ids, 3);

                currentState = SM1Thread.States.SimpleState0;
                return true;
            }

            private boolean t_NewState1_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                if(i > 5) {
                    lock_ids[0] = target_locks[0]; // Release i
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            private boolean t_NewState1_0_s_0_n_1() {
                if(i < 0) {
                    lock_ids[0] = target_locks[0]; // Release i
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState1 -> NewState0
            private boolean execute_transition_NewState1_0() {
                // SLCO composite [!(i > 5 or i < 0); i := y[i]; x[y[i]] := 1] -> [!(i > 5 or i < 0); i := y[i]; x[y[i]] := 1] 
                if(!(!((t_NewState1_0_s_0_n_0() || t_NewState1_0_s_0_n_1())))) {
                    return false;
                }
                lock_ids[0] = target_locks[3] = 4 + 0; // Acquire y[0]
                lock_ids[1] = target_locks[2] = 4 + 1; // Acquire y[1]
                lock_ids[2] = target_locks[1] = 4 + i; // Acquire y[i]
                lockManager.acquire_locks(lock_ids, 3);
                lock_ids[0] = target_locks[4] = 4 + y[i]; // Acquire y[y[i]]
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[3]; // Release y[0]
                lock_ids[1] = target_locks[2]; // Release y[1]
                lockManager.release_locks(lock_ids, 2);
                i = y[i];
                lock_ids[0] = target_locks[1]; // Release y[i]
                lockManager.release_locks(lock_ids, 1);
                lock_ids[0] = target_locks[5] = 6 + y[i]; // Acquire x[y[i]]
                lockManager.acquire_locks(lock_ids, 1);
                x[y[i]] = 1;
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[5]; // Release x[y[i]]
                lock_ids[2] = target_locks[4]; // Release y[y[i]]
                lockManager.release_locks(lock_ids, 3);

                currentState = SM1Thread.States.NewState0;
                return true;
            }

            private boolean t_NewState2_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            private boolean t_NewState2_0_s_0_n_1() {
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            private boolean t_NewState2_0_s_0_n_2() {
                lock_ids[0] = target_locks[1] = 1 + i; // Acquire b[i]
                lockManager.acquire_locks(lock_ids, 1);
                if(b[i]) {
                    lock_ids[0] = target_locks[1]; // Release b[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release b[i]
                lock_ids[1] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState2 -> NewState1
            private boolean execute_transition_NewState2_0() {
                // SLCO composite [i >= 0 and i < 2 and b[i]; x[i] := x[i] + 1; i := 0] -> [i >= 0 and i < 2 and b[i]; x[i] := x[i] + 1; i := 0] 
                if(!(t_NewState2_0_s_0_n_0() && t_NewState2_0_s_0_n_1() && t_NewState2_0_s_0_n_2())) {
                    return false;
                }
                lock_ids[0] = target_locks[2] = 6 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                x[i] = x[i] + 1;
                lock_ids[0] = target_locks[2]; // Release x[i]
                lockManager.release_locks(lock_ids, 1);
                i = 0;
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.NewState1;
                return true;
            }

            private boolean t_NewState3_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                if(i >= 0) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            private boolean t_NewState3_0_s_0_n_1() {
                if(i < 2) {
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            private boolean t_NewState3_0_s_0_n_2() {
                lock_ids[0] = target_locks[1] = 1 + i; // Acquire b[i]
                lockManager.acquire_locks(lock_ids, 1);
                if(b[i]) {
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release b[i]
                lock_ids[1] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            private boolean t_NewState3_0_s_2_n_3() {
                if(x[i] > 5) {
                    lock_ids[0] = target_locks[2]; // Release x[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release x[i]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState3 -> NewState2
            private boolean execute_transition_NewState3_0() {
                // SLCO composite [i >= 0 and i < 2 and b[i]; x[i] := x[i] + 1; b[i] := b[i] xor x[i] > 5] -> [i >= 0 and i < 2 and b[i]; x[i] := x[i] + 1; b[i] := b[i] xor x[i] > 5] 
                if(!(t_NewState3_0_s_0_n_0() && t_NewState3_0_s_0_n_1() && t_NewState3_0_s_0_n_2())) {
                    return false;
                }
                lock_ids[0] = target_locks[2] = 6 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                x[i] = x[i] + 1;
                b[i] = b[i] xor t_NewState3_0_s_2_n_3();
                lock_ids[0] = target_locks[1]; // Release b[i]
                lock_ids[1] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 2);

                currentState = SM1Thread.States.NewState2;
                return true;
            }

            private boolean t_NewState4_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                return i < 0;
            }

            private boolean t_NewState4_0_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 1 + i; // Acquire b[i]
                lockManager.acquire_locks(lock_ids, 1);
                if(b[i]) {
                    lock_ids[0] = target_locks[1]; // Release b[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release b[i]
                lock_ids[1] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState4 -> NewState3
            private boolean execute_transition_NewState4_0() {
                // SLCO composite [i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; i := 0] -> [i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; i := 0] 
                if(!(t_NewState4_0_s_0_n_0() || i >= 2 || t_NewState4_0_s_0_n_1())) {
                    return false;
                }
                lock_ids[0] = target_locks[2] = 6 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                x[i] = x[i] + 1;
                lock_ids[0] = target_locks[2]; // Release x[i]
                lockManager.release_locks(lock_ids, 1);
                i = 0;
                lock_ids[0] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.NewState3;
                return true;
            }

            private boolean t_NewState5_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                return i < 0;
            }

            private boolean t_NewState5_0_s_0_n_1() {
                if(t_NewState5_0_s_0_n_0() || i >= 2) {
                    lock_ids[0] = target_locks[1] = 1 + i; // Acquire b[i]
                    lockManager.acquire_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            private boolean t_NewState5_0_s_0_n_2() {
                lock_ids[0] = target_locks[1] = 1 + i; // Acquire b[i]
                lockManager.acquire_locks(lock_ids, 1);
                if(b[i]) {
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release b[i]
                lock_ids[1] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            private boolean t_NewState5_0_s_2_n_3() {
                if(!(b[i])) {
                    lock_ids[0] = target_locks[2]; // Release x[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            private boolean t_NewState5_0_s_2_n_4() {
                if(x[i] > 5) {
                    lock_ids[0] = target_locks[2]; // Release x[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release x[i]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState5 -> NewState4
            private boolean execute_transition_NewState5_0() {
                // SLCO composite [i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; b[i] := !b[i] or x[i] > 5] -> [i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; b[i] := !b[i] or x[i] > 5] 
                if(!(t_NewState5_0_s_0_n_1() || t_NewState5_0_s_0_n_2())) {
                    return false;
                }
                lock_ids[0] = target_locks[2] = 6 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                x[i] = x[i] + 1;
                b[i] = t_NewState5_0_s_2_n_3() || t_NewState5_0_s_2_n_4();
                lock_ids[0] = target_locks[1]; // Release b[i]
                lock_ids[1] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 2);

                currentState = SM1Thread.States.NewState4;
                return true;
            }

            private boolean t_NewState6_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                return i < 0;
            }

            private boolean t_NewState6_0_s_0_n_1() {
                if(t_NewState6_0_s_0_n_0() || i >= 2) {
                    lock_ids[0] = target_locks[1] = 1 + i; // Acquire b[i]
                    lockManager.acquire_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            private boolean t_NewState6_0_s_0_n_2() {
                lock_ids[0] = target_locks[1] = 1 + i; // Acquire b[i]
                lockManager.acquire_locks(lock_ids, 1);
                if(b[i]) {
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release b[i]
                lock_ids[1] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 2);
                return false;
            }

            private boolean t_NewState6_0_s_2_n_3() {
                if(!(b[i])) {
                    lock_ids[0] = target_locks[2]; // Release x[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            private boolean t_NewState6_0_s_2_n_4() {
                if(x[i] > 5) {
                    lock_ids[0] = target_locks[2]; // Release x[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release x[i]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState6 -> NewState5
            private boolean execute_transition_NewState6_0() {
                // SLCO composite [i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; b[i] := !b[i] or x[i] > 5] -> [i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; b[i] := !b[i] or x[i] > 5] 
                if(!(t_NewState6_0_s_0_n_1() || t_NewState6_0_s_0_n_2())) {
                    return false;
                }
                lock_ids[0] = target_locks[2] = 6 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                x[i] = x[i] + 1;
                b[i] = t_NewState6_0_s_2_n_3() || t_NewState6_0_s_2_n_4();
                lock_ids[0] = target_locks[1]; // Release b[i]
                lock_ids[1] = target_locks[0]; // Release i
                lockManager.release_locks(lock_ids, 2);

                currentState = SM1Thread.States.NewState5;
                return true;
            }

            private boolean t_NewState7_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 1 + 0; // Acquire b[0]
                lockManager.acquire_locks(lock_ids, 1);
                if(b[0]) {
                    lock_ids[0] = target_locks[1] = 1 + 1; // Acquire b[1]
                    lockManager.acquire_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            private boolean t_NewState7_0_s_0_n_1() {
                lock_ids[0] = target_locks[1] = 1 + 1; // Acquire b[1]
                lockManager.acquire_locks(lock_ids, 1);
                return b[1];
            }

            private boolean t_NewState7_0_s_0_n_2() {
                if(t_NewState7_0_s_0_n_0() || t_NewState7_0_s_0_n_1()) {
                    lock_ids[0] = target_locks[2] = 1 + 2; // Acquire b[2]
                    lockManager.acquire_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            private boolean t_NewState7_0_s_0_n_3() {
                lock_ids[0] = target_locks[2] = 1 + 2; // Acquire b[2]
                lockManager.acquire_locks(lock_ids, 1);
                if(b[2]) {
                    return true;
                }
                lock_ids[0] = target_locks[2]; // Release b[2]
                lock_ids[1] = target_locks[0]; // Release b[0]
                lock_ids[2] = target_locks[1]; // Release b[1]
                lockManager.release_locks(lock_ids, 3);
                return false;
            }

            private boolean t_NewState7_0_s_3_n_4() {
                if(b[0]) {
                    lock_ids[0] = target_locks[1]; // Release b[1]
                    lock_ids[1] = target_locks[0]; // Release b[0]
                    lockManager.release_locks(lock_ids, 2);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release b[0]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            private boolean t_NewState7_0_s_3_n_5() {
                if(b[1]) {
                    lock_ids[0] = target_locks[1]; // Release b[1]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release b[1]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState7 -> NewState6
            private boolean execute_transition_NewState7_0() {
                // SLCO composite [b[0] or b[1] or b[2]; b[0] := true; b[1] := false; b[2] := b[0] or b[1] or b[2]] -> [b[0] or b[1] or b[2]; b[0] := true; b[1] := false; b[2] := b[0] or b[1] or b[2]] 
                if(!(t_NewState7_0_s_0_n_2() || t_NewState7_0_s_0_n_3())) {
                    return false;
                }
                b[0] = true;
                b[1] = false;
                b[2] = t_NewState7_0_s_3_n_4() || t_NewState7_0_s_3_n_5() || b[2];
                lock_ids[0] = target_locks[2]; // Release b[2]
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.NewState6;
                return true;
            }

            private boolean t_NewState8_0_s_0_n_0() {
                lock_ids[0] = target_locks[2] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[1] = 1 + 0; // Acquire b[0]
                lock_ids[1] = target_locks[3] = 1 + 1; // Acquire b[1]
                lock_ids[2] = target_locks[4] = 1 + 2; // Acquire b[2]
                lock_ids[3] = target_locks[0] = 1 + i; // Acquire b[i]
                lockManager.acquire_locks(lock_ids, 4);
                if(b[0]) {
                    lock_ids[0] = target_locks[1]; // Release b[0]
                    lock_ids[1] = target_locks[3]; // Release b[1]
                    lock_ids[2] = target_locks[4]; // Release b[2]
                    lockManager.release_locks(lock_ids, 3);
                    return true;
                }
                return false;
            }

            private boolean t_NewState8_0_s_0_n_1() {
                if(i < 0) {
                    lock_ids[0] = target_locks[1]; // Release b[0]
                    lock_ids[1] = target_locks[3]; // Release b[1]
                    lock_ids[2] = target_locks[4]; // Release b[2]
                    lockManager.release_locks(lock_ids, 3);
                    return true;
                }
                return false;
            }

            private boolean t_NewState8_0_s_0_n_2() {
                if(i >= 2) {
                    lock_ids[0] = target_locks[1]; // Release b[0]
                    lock_ids[1] = target_locks[3]; // Release b[1]
                    lock_ids[2] = target_locks[4]; // Release b[2]
                    lockManager.release_locks(lock_ids, 3);
                    return true;
                }
                return false;
            }

            private boolean t_NewState8_0_s_0_n_3() {
                if(b[i]) {
                    lock_ids[0] = target_locks[1]; // Release b[0]
                    lock_ids[1] = target_locks[3]; // Release b[1]
                    lock_ids[2] = target_locks[4]; // Release b[2]
                    lockManager.release_locks(lock_ids, 3);
                    return true;
                }
                lock_ids[0] = target_locks[1]; // Release b[0]
                lock_ids[1] = target_locks[3]; // Release b[1]
                lock_ids[2] = target_locks[4]; // Release b[2]
                lock_ids[3] = target_locks[0]; // Release b[i]
                lock_ids[4] = target_locks[2]; // Release i
                lockManager.release_locks(lock_ids, 5);
                return false;
            }

            private boolean t_NewState8_0_s_2_n_4() {
                if(!(b[i])) {
                    lock_ids[0] = target_locks[5]; // Release x[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            private boolean t_NewState8_0_s_2_n_5() {
                if(x[i] > 5) {
                    lock_ids[0] = target_locks[5]; // Release x[i]
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[5]; // Release x[i]
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (id:0, p:0) | NewState8 -> NewState7
            private boolean execute_transition_NewState8_0() {
                // SLCO composite [b[0] or i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; b[i] := !b[i] or x[i] > 5] -> [b[0] or i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; b[i] := !b[i] or x[i] > 5] 
                if(!(t_NewState8_0_s_0_n_0() || t_NewState8_0_s_0_n_1() || t_NewState8_0_s_0_n_2() || t_NewState8_0_s_0_n_3())) {
                    return false;
                }
                lock_ids[0] = target_locks[5] = 6 + i; // Acquire x[i]
                lockManager.acquire_locks(lock_ids, 1);
                x[i] = x[i] + 1;
                b[i] = t_NewState8_0_s_2_n_4() || t_NewState8_0_s_2_n_5();
                lock_ids[0] = target_locks[0]; // Release b[i]
                lock_ids[1] = target_locks[2]; // Release i
                lockManager.release_locks(lock_ids, 2);

                currentState = SM1Thread.States.NewState7;
                return true;
            }

            // SLCO transition (id:0, p:0) | SimpleState0 -> NewState8
            private boolean execute_transition_SimpleState0_0() {
                // SLCO expression !(true) -> !(true) (Superfluous)
                if(!(true)) {
                    return false;
                }

                // SLCO composite [i := 0; x[y[i]] := 1; x[0] := 1] -> [true; i := 0; x[y[i]] := 1; x[0] := 1] 
                if(!(true)) {
                    return true;
                }
                lock_ids[0] = target_locks[0] = 0; // Acquire i
                lockManager.acquire_locks(lock_ids, 1);
                i = 0;
                lock_ids[0] = target_locks[3] = 4 + i; // Acquire y[i]
                lockManager.acquire_locks(lock_ids, 1);
                lock_ids[0] = target_locks[1] = 6 + y[i]; // Acquire x[y[i]]
                lock_ids[1] = target_locks[2] = 6 + 0; // Acquire x[0]
                lockManager.acquire_locks(lock_ids, 2);
                x[y[i]] = 1;
                lock_ids[0] = target_locks[0]; // Release i
                lock_ids[1] = target_locks[1]; // Release x[y[i]]
                lock_ids[2] = target_locks[3]; // Release y[i]
                lockManager.release_locks(lock_ids, 3);
                x[0] = 1;
                lock_ids[0] = target_locks[2]; // Release x[0]
                lockManager.release_locks(lock_ids, 1);

                currentState = SM1Thread.States.NewState8;
                return true;
            }

            private void exec_SMC0() {
                // There are no transitions starting in state SMC0.
            }

            private void exec_SMC1() {
                // There are no transitions starting in state SMC1.
            }

            private void exec_NewState0() {
                // SLCO transition (id:0, p:0) | NewState0 -> SimpleState0 | guard: [i > 5; i := y[i]; x[y[i]] := 1]
                if(execute_transition_NewState0_0()) {
                    return;
                }
            }

            private void exec_NewState1() {
                // SLCO transition (id:0, p:0) | NewState1 -> NewState0 | guard: [!(i > 5 or i < 0); i := y[i]; x[y[i]] := 1]
                if(execute_transition_NewState1_0()) {
                    return;
                }
            }

            private void exec_NewState2() {
                // SLCO transition (id:0, p:0) | NewState2 -> NewState1 | guard: [i >= 0 and i < 2 and b[i]; x[i] := x[i] + 1; i := 0]
                if(execute_transition_NewState2_0()) {
                    return;
                }
            }

            private void exec_NewState3() {
                // SLCO transition (id:0, p:0) | NewState3 -> NewState2 | guard: [i >= 0 and i < 2 and b[i]; x[i] := x[i] + 1; b[i] := b[i] xor x[i] > 5]
                if(execute_transition_NewState3_0()) {
                    return;
                }
            }

            private void exec_NewState4() {
                // SLCO transition (id:0, p:0) | NewState4 -> NewState3 | guard: [i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; i := 0]
                if(execute_transition_NewState4_0()) {
                    return;
                }
            }

            private void exec_NewState5() {
                // SLCO transition (id:0, p:0) | NewState5 -> NewState4 | guard: [i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; b[i] := !b[i] or x[i] > 5]
                if(execute_transition_NewState5_0()) {
                    return;
                }
            }

            private void exec_NewState6() {
                // SLCO transition (id:0, p:0) | NewState6 -> NewState5 | guard: [i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; b[i] := !b[i] or x[i] > 5]
                if(execute_transition_NewState6_0()) {
                    return;
                }
            }

            private void exec_NewState7() {
                // SLCO transition (id:0, p:0) | NewState7 -> NewState6 | guard: [b[0] or b[1] or b[2]; b[0] := true; b[1] := false; b[2] := b[0] or b[1] or b[2]]
                if(execute_transition_NewState7_0()) {
                    return;
                }
            }

            private void exec_NewState8() {
                // SLCO transition (id:0, p:0) | NewState8 -> NewState7 | guard: [b[0] or i < 0 or i >= 2 or b[i]; x[i] := x[i] + 1; b[i] := !b[i] or x[i] > 5]
                if(execute_transition_NewState8_0()) {
                    return;
                }
            }

            private void exec_SimpleState0() {
                // SLCO transition (id:0, p:0) | SimpleState0 -> NewState8 | guard: true
                if(execute_transition_SimpleState0_0()) {
                    return;
                }
            }

            // Execute method
            private void exec() {
                while(true) {
                    switch(currentState) {
                        case SMC0 -> exec_SMC0();
                        case SMC1 -> exec_SMC1();
                        case NewState0 -> exec_NewState0();
                        case NewState1 -> exec_NewState1();
                        case NewState2 -> exec_NewState2();
                        case NewState3 -> exec_NewState3();
                        case NewState4 -> exec_NewState4();
                        case NewState5 -> exec_NewState5();
                        case NewState6 -> exec_NewState6();
                        case NewState7 -> exec_NewState7();
                        case NewState8 -> exec_NewState8();
                        case SimpleState0 -> exec_SimpleState0();
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
            LockManager lockManager = new LockManager(8);

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