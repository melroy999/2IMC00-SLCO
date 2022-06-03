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
            new P()
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

        P() {
            // Create a lock manager.
            LockManager lockManager = new LockManager(0);
            // Instantiate the state machine threads and pass on the class' lock manager.
            T_SM1 = new P_SM1Thread(lockManager);
            T_SM2 = new P_SM2Thread(lockManager);
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

            // Thread local variables.
            private int a;
            private int b;
            private int c;
            private int d;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            P_SM1Thread(LockManager lockManagerInstance) {
                currentState = P_SM1Thread.States.SMC0;
                lockManager = lockManagerInstance;
                lock_ids = new int[0];
                target_locks = new int[0];
                random = new Random();

                // Variable instantiations.
                a = 0;
                b = 0;
                c = 0;
                d = 0;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | a > 10.
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | a > 10.
                if(!(a > 10)) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | a > 11.
            private boolean execute_transition_SMC0_1() {
                // SLCO expression | a > 11.
                if(!(a > 11)) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | a > 13 and a < 17.
            private boolean execute_transition_SMC0_2() {
                // SLCO expression | a > 13 and a < 17.
                if(!(a > 13 && a < 17)) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:3) | SMC0 -> SMC0 | a > 11 and a < 15.
            private boolean execute_transition_SMC0_3() {
                // SLCO expression | a > 11 and a < 15.
                if(!(a > 11 && a < 15)) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:4) | SMC0 -> SMC0 | a > 11 and a < 13.
            private boolean execute_transition_SMC0_4() {
                // SLCO expression | a > 11 and a < 13.
                if(!(a > 11 && a < 13)) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:5) | SMC0 -> SMC0 | a > 13 and a < 15.
            private boolean execute_transition_SMC0_5() {
                // SLCO expression | a > 13 and a < 15.
                if(!(a > 13 && a < 15)) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:6) | SMC0 -> SMC0 | a > 15 and a < 20.
            private boolean execute_transition_SMC0_6() {
                // SLCO expression | a > 15 and a < 20.
                if(!(a > 15 && a < 20)) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:7) | SMC0 -> SMC0 | a > 15 and a < 17.
            private boolean execute_transition_SMC0_7() {
                // SLCO expression | a > 15 and a < 17.
                if(!(a > 15 && a < 17)) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:8) | SMC0 -> SMC0 | a > 17 and a < 20.
            private boolean execute_transition_SMC0_8() {
                // SLCO expression | a > 17 and a < 20.
                if(!(a > 17 && a < 20)) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:9) | SMC0 -> SMC0 | a < 1.
            private boolean execute_transition_SMC0_9() {
                // SLCO expression | a < 1.
                if(!(a < 1)) {
                    return false;
                }

                currentState = P_SM1Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:10) | SMC0 -> SMC0 | a < 2.
            private boolean execute_transition_SMC0_10() {
                // SLCO expression | a < 2.
                if(!(a < 2)) {
                    return false;
                }

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
                // SLCO transition (p:0, id:3) | SMC0 -> SMC0 | a > 11 and a < 15.
                if(execute_transition_SMC0_3()) {
                    return;
                }
                // [DET.START]
                // SLCO transition (p:0, id:4) | SMC0 -> SMC0 | a > 11 and a < 13.
                if(execute_transition_SMC0_4()) {
                    return;
                }
                // SLCO transition (p:0, id:5) | SMC0 -> SMC0 | a > 13 and a < 15.
                if(execute_transition_SMC0_5()) {
                    return;
                }
                // [DET.END]
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
                // SLCO transition (p:0, id:9) | SMC0 -> SMC0 | a < 1.
                if(execute_transition_SMC0_9()) {
                    return;
                }
                // SLCO transition (p:0, id:10) | SMC0 -> SMC0 | a < 2.
                if(execute_transition_SMC0_10()) {
                    return;
                }
                // [SEQ.END]
                // [DET.END]
                // [DET.START]
                // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | a > 13 and a < 17.
                if(execute_transition_SMC0_2()) {
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

            // Thread local variables.
            private int a;
            private int b;
            private int c;
            private int d;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            P_SM2Thread(LockManager lockManagerInstance) {
                currentState = P_SM2Thread.States.SMC0;
                lockManager = lockManagerInstance;
                lock_ids = new int[0];
                target_locks = new int[0];
                random = new Random();

                // Variable instantiations.
                a = 0;
                b = 0;
                c = 0;
                d = 0;
            }

            // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | a > 10.
            private boolean execute_transition_SMC0_0() {
                // SLCO expression | a > 10.
                if(!(a > 10)) {
                    return false;
                }

                currentState = P_SM2Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:1) | SMC0 -> SMC0 | a < 1.
            private boolean execute_transition_SMC0_1() {
                // SLCO expression | a < 1.
                if(!(a < 1)) {
                    return false;
                }

                currentState = P_SM2Thread.States.SMC0;
                return true;
            }

            // SLCO transition (p:0, id:2) | SMC0 -> SMC0 | a < 1.
            private boolean execute_transition_SMC0_2() {
                // SLCO expression | a < 1.
                if(!(a < 1)) {
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