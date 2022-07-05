package testing;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.time.Duration;
import java.time.Instant;

// SLCO model Nesting.
public class Nesting {
    // The objects in the model.
    private final SLCO_Class[] objects;

    // Interface for SLCO classes.
    interface SLCO_Class {
        void startThreads();
        void joinThreads();
    }

    Nesting() {
        // Instantiate the objects.
        objects = new SLCO_Class[] {
            new P(0)
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
        private final Thread T_SM2;
        private final Thread T_SM3;

        // Class variables.
        private volatile int a;

        P(int a) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(1);

            // Instantiate the class variables.
            this.a = a;

            // Instantiate the state machine threads and pass on the class' lock manager.
            T_SM1 = new P_SM1Thread(lockManager);
            T_SM2 = new P_SM2Thread(lockManager);
            T_SM3 = new P_SM3Thread(lockManager);
        }

        // Define the states fot the state machine SM1.
        interface P_SM1Thread_States {
            enum States {
                SMC0
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
                target_locks = new int[1];
                random = new Random();
            }

            // SLCO expression wrapper | a > 10.
            private boolean t_SMC0_0_s_0_n_0() {
                if(a > 10) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | a > 10.
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | a > 10.
                if(!(t_SMC0_0_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a > 11.
            private boolean t_SMC0_1_s_0_n_0() {
                if(a > 11) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | a > 11.
            private boolean execute_transition_SMC0_1() {
                // SLCO expression | a > 11.
                if(!(t_SMC0_1_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 17.
            private boolean t_SMC0_2_s_0_n_0() {
                if(a < 17) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | a > 13 and a < 17.
            private boolean execute_transition_SMC0_2() {
                // SLCO expression | a > 13 and a < 17.
                if(!(a > 13 && t_SMC0_2_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 13.
            private boolean t_SMC0_3_s_0_n_0() {
                if(a < 13) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:3) | SMC0 -> SMC0 | a > 11 and a < 13.
            private boolean execute_transition_SMC0_3() {
                // SLCO expression | a > 11 and a < 13.
                if(!(a > 11 && t_SMC0_3_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 15.
            private boolean t_SMC0_4_s_0_n_0() {
                if(a < 15) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:4) | SMC0 -> SMC0 | a > 13 and a < 15.
            private boolean execute_transition_SMC0_4() {
                // SLCO expression | a > 13 and a < 15.
                if(!(a > 13 && t_SMC0_4_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 15.
            private boolean t_SMC0_5_s_0_n_0() {
                if(a < 15) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:5) | SMC0 -> SMC0 | a > 13 and a < 15.
            private boolean execute_transition_SMC0_5() {
                // SLCO expression | a > 13 and a < 15.
                if(!(a > 13 && t_SMC0_5_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 20.
            private boolean t_SMC0_6_s_0_n_0() {
                if(a < 20) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:6) | SMC0 -> SMC0 | a > 15 and a < 20.
            private boolean execute_transition_SMC0_6() {
                // SLCO expression | a > 15 and a < 20.
                if(!(a > 15 && t_SMC0_6_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 17.
            private boolean t_SMC0_7_s_0_n_0() {
                if(a < 17) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:7) | SMC0 -> SMC0 | a > 15 and a < 17.
            private boolean execute_transition_SMC0_7() {
                // SLCO expression | a > 15 and a < 17.
                if(!(a > 15 && t_SMC0_7_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 20.
            private boolean t_SMC0_8_s_0_n_0() {
                if(a < 20) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:8) | SMC0 -> SMC0 | a > 17 and a < 20.
            private boolean execute_transition_SMC0_8() {
                // SLCO expression | a > 17 and a < 20.
                if(!(a > 17 && t_SMC0_8_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 15.
            private boolean t_SMC0_9_s_0_n_0() {
                if(a < 15) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:9) | SMC0 -> SMC0 | a > 11 and a < 15.
            private boolean execute_transition_SMC0_9() {
                // SLCO expression | a > 11 and a < 15.
                if(!(a > 11 && t_SMC0_9_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 15.
            private boolean t_SMC0_10_s_0_n_0() {
                if(a < 15) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:10) | SMC0 -> SMC0 | a > 11 and a < 15.
            private boolean execute_transition_SMC0_10() {
                // SLCO expression | a > 11 and a < 15.
                if(!(a > 11 && t_SMC0_10_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 1.
            private boolean t_SMC0_11_s_0_n_0() {
                if(a < 1) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:11) | SMC0 -> SMC0 | a < 1.
            private boolean execute_transition_SMC0_11() {
                // SLCO expression | a < 1.
                if(!(t_SMC0_11_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 1.
            private boolean t_SMC0_12_s_0_n_0() {
                if(a < 1) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:12) | SMC0 -> SMC0 | a < 1.
            private boolean execute_transition_SMC0_12() {
                // SLCO expression | a < 1.
                if(!(t_SMC0_12_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 2.
            private boolean t_SMC0_13_s_0_n_0() {
                if(a < 2) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:13) | SMC0 -> SMC0 | a < 2.
            private boolean execute_transition_SMC0_13() {
                // SLCO expression | a < 2.
                if(!(t_SMC0_13_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 2.
            private boolean t_SMC0_14_s_0_n_0() {
                if(a < 2) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release a
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:2, id:14) | SMC0 -> SMC0 | a < 2.
            private boolean execute_transition_SMC0_14() {
                // SLCO expression | a < 2.
                if(!(t_SMC0_14_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | true.
            private boolean t_SMC0_15_s_0_n_0() {
                lock_ids[0] = target_locks[0]; // Release a
                lockManager.release_locks(lock_ids, 1);
                return true;
            }

            // SLCO transition (p:0, id:15) | SMC0 -> SMC0 | true | a := 2.
            private boolean execute_transition_SMC0_15() {
                // SLCO expression | true.
                if(!(t_SMC0_15_s_0_n_0())) {
                    return false;
                }

                // SLCO assignment | a := 2.
                lock_ids[0] = target_locks[0] = 0; // Acquire a
                lockManager.acquire_locks(lock_ids, 1);
                a = 2;
                lock_ids[0] = target_locks[0]; // Release a
                lockManager.release_locks(lock_ids, 1);

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // Attempt to fire a transition starting in state SMC0.
            private void exec_SMC0() {
                // [SEQ.START]
                // [DET.START]
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | a > 10.
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | a > 11.
                if(execute_transition_SMC0_1()) {
                    return;
                }
                // [DET.START]
                // [SEQ.START]
                // [DET.START]
                // SLCO transition (p:0, id:3) | SMC0 -> SMC0 | a > 11 and a < 13.
                if(execute_transition_SMC0_3()) {
                    return;
                }
                // [SEQ.START]
                // SLCO transition (p:0, id:4) | SMC0 -> SMC0 | a > 13 and a < 15.
                if(execute_transition_SMC0_4()) {
                    return;
                }
                // SLCO transition (p:0, id:5) | SMC0 -> SMC0 | a > 13 and a < 15.
                if(execute_transition_SMC0_5()) {
                    return;
                }
                // [SEQ.END]
                // [DET.END]
                // SLCO transition (p:0, id:9) | SMC0 -> SMC0 | a > 11 and a < 15.
                if(execute_transition_SMC0_9()) {
                    return;
                }
                // SLCO transition (p:0, id:10) | SMC0 -> SMC0 | a > 11 and a < 15.
                if(execute_transition_SMC0_10()) {
                    return;
                }
                // [SEQ.END]
                // [SEQ.START]
                // SLCO transition (p:0, id:6) | SMC0 -> SMC0 | a > 15 and a < 20.
                if(execute_transition_SMC0_6()) {
                    return;
                }
                // [DET.START]
                // SLCO transition (p:0, id:7) | SMC0 -> SMC0 | a > 15 and a < 17.
                if(execute_transition_SMC0_7()) {
                    return;
                }
                // SLCO transition (p:0, id:8) | SMC0 -> SMC0 | a > 17 and a < 20.
                if(execute_transition_SMC0_8()) {
                    return;
                }
                // [DET.END]
                // [SEQ.END]
                // [DET.END]
                // [SEQ.END]
                // [SEQ.START]
                // SLCO transition (p:0, id:11) | SMC0 -> SMC0 | a < 1.
                if(execute_transition_SMC0_11()) {
                    return;
                }
                // SLCO transition (p:0, id:12) | SMC0 -> SMC0 | a < 1.
                if(execute_transition_SMC0_12()) {
                    return;
                }
                // SLCO transition (p:0, id:13) | SMC0 -> SMC0 | a < 2.
                if(execute_transition_SMC0_13()) {
                    return;
                }
                // [SEQ.END]
                // [DET.END]
                // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | a > 13 and a < 17.
                if(execute_transition_SMC0_2()) {
                    return;
                }
                // SLCO transition (p:0, id:15) | SMC0 -> SMC0 | true | a := 2.
                if(execute_transition_SMC0_15()) {
                    return;
                }
                // SLCO transition (p:2, id:14) | SMC0 -> SMC0 | a < 2.
                if(execute_transition_SMC0_14()) {
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

        // Define the states fot the state machine SM2.
        interface P_SM2Thread_States {
            enum States {
                SMC0
            }
        }

        // Representation of the SLCO state machine SM2.
        class P_SM2Thread extends Thread implements P_SM2Thread_States {
            // Current state
            private P_SM2Thread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            P_SM2Thread(LockManager lockManagerInstance) {
                currentState = P_SM2Thread.States.SMC0;
                lockManager = lockManagerInstance;
                lock_ids = new int[1];
                target_locks = new int[1];
                random = new Random();
            }

            // SLCO expression wrapper | a > 10.
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire a
                lockManager.acquire_locks(lock_ids, 1);
                if(a > 10) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | a > 10.
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | a > 10.
                if(!(t_SMC0_0_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM2Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 1.
            private boolean t_SMC0_1_s_0_n_0() {
                if(a < 1) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                return false;
            }

            // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | a < 1.
            private boolean execute_transition_SMC0_1() {
                // SLCO expression | a < 1.
                if(!(t_SMC0_1_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM2Thread.States.SMC0;
                return true;
            }

            // SLCO expression wrapper | a < 1.
            private boolean t_SMC0_2_s_0_n_0() {
                if(a < 1) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release a
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | a < 1.
            private boolean execute_transition_SMC0_2() {
                // SLCO expression | a < 1.
                if(!(t_SMC0_2_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM2Thread.States.SMC0;
                return true;
            }

            // Attempt to fire a transition starting in state SMC0.
            private void exec_SMC0() {
                // [SEQ.START]
                // [DET.START]
                // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | a > 10.
                if(execute_transition_SMC0_0()) {
                    return;
                }
                // [SEQ.START]
                // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | a < 1.
                if(execute_transition_SMC0_1()) {
                    return;
                }
                // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | a < 1.
                if(execute_transition_SMC0_2()) {
                    return;
                }
                // [SEQ.END]
                // [DET.END]
                // [SEQ.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 60) {
                    switch(currentState) {
                        case SMC0 -> exec_SMC0();
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

        // Define the states fot the state machine SM3.
        interface P_SM3Thread_States {
            enum States {
                SMC0
            }
        }

        // Representation of the SLCO state machine SM3.
        class P_SM3Thread extends Thread implements P_SM3Thread_States {
            // Current state
            private P_SM3Thread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            P_SM3Thread(LockManager lockManagerInstance) {
                currentState = P_SM3Thread.States.SMC0;
                lockManager = lockManagerInstance;
                lock_ids = new int[1];
                target_locks = new int[1];
                random = new Random();
            }

            // SLCO expression wrapper | a > 10.
            private boolean t_SMC0_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire a
                lockManager.acquire_locks(lock_ids, 1);
                if(a > 10) {
                    lock_ids[0] = target_locks[0]; // Release a
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release a
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | a > 10.
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | a > 10.
                if(!(t_SMC0_0_s_0_n_0())) {
                    return false;
                }

                currentState = P_SM3Thread.States.SMC0;
                return true;
            }

            // Attempt to fire a transition starting in state SMC0.
            private void exec_SMC0() {
                // [SEQ.START]
                // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | a > 10.
                if(execute_transition_SMC0_0()) {
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
            T_SM2.start();
            T_SM3.start();
        }

        // Join all threads.
        public void joinThreads() {
            while (true) {
                try {
                    T_SM1.join();
                    T_SM2.join();
                    T_SM3.join();
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
        Nesting model = new Nesting();
        model.startThreads();
        model.joinThreads();
    }
}