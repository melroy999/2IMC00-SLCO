package testing.randompicknostructures;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.time.Duration;
import java.time.Instant;

// SLCO model CounterDistributor.
public class CounterDistributor {
    // The objects in the model.
    private final SLCO_Class[] objects;

    // Interface for SLCO classes.
    interface SLCO_Class {
        void startThreads();
        void joinThreads();
    }

    CounterDistributor() {
        // Instantiate the objects.
        objects = new SLCO_Class[] {
            new CounterDistributorExact(0)
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

    // Representation of the SLCO class CounterDistributorExact.
    private static class CounterDistributorExact implements SLCO_Class {
        // The state machine threads.
        private final Thread T_Counter;
        private final Thread T_Distributor;

        // Class variables.
        private volatile int x;

        CounterDistributorExact(int x) {
            // Create a lock manager.
            LockManager lockManager = new LockManager(1);

            // Instantiate the class variables.
            this.x = x;

            // Instantiate the state machine threads and pass on the class' lock manager.
            T_Counter = new CounterDistributorExact_CounterThread(lockManager);
            T_Distributor = new CounterDistributorExact_DistributorThread(lockManager);
        }

        // Define the states fot the state machine Counter.
        interface CounterDistributorExact_CounterThread_States {
            enum States {
                C
            }
        }

        // Representation of the SLCO state machine Counter.
        class CounterDistributorExact_CounterThread extends Thread implements CounterDistributorExact_CounterThread_States {
            // Current state
            private CounterDistributorExact_CounterThread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            CounterDistributorExact_CounterThread(LockManager lockManagerInstance) {
                currentState = CounterDistributorExact_CounterThread.States.C;
                lockManager = lockManagerInstance;
                lock_ids = new int[1];
                target_locks = new int[1];
                random = new Random();
            }

            // SLCO transition (p:0, id:0) | C -> C | true | x := (x + 1) % 10.
            private boolean execute_transition_C_0() {
                // (Superfluous) SLCO expression | true.

                // SLCO assignment | x := (x + 1) % 10.
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                x = Math.floorMod((x + 1), 10);
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);

                currentState = CounterDistributorExact_CounterThread.States.C;
                return true;
            }

            // Attempt to fire a transition starting in state C.
            private void exec_C() {
                // [N_DET.START]
                // SLCO transition (p:0, id:0) | C -> C | true | x := (x + 1) % 10.
                if(execute_transition_C_0()) {
                    return;
                }
                // [N_DET.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 30) {
                    switch(currentState) {
                        case C -> exec_C();
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

        // Define the states fot the state machine Distributor.
        interface CounterDistributorExact_DistributorThread_States {
            enum States {
                P
            }
        }

        // Representation of the SLCO state machine Distributor.
        class CounterDistributorExact_DistributorThread extends Thread implements CounterDistributorExact_DistributorThread_States {
            // Current state
            private CounterDistributorExact_DistributorThread.States currentState;

            // Random number generator to handle non-determinism.
            private final Random random;

            // The lock manager of the parent class.
            private final LockManager lockManager;

            // A list of lock ids and target locks that can be reused.
            private final int[] lock_ids;
            private final int[] target_locks;

            CounterDistributorExact_DistributorThread(LockManager lockManagerInstance) {
                currentState = CounterDistributorExact_DistributorThread.States.P;
                lockManager = lockManagerInstance;
                lock_ids = new int[1];
                target_locks = new int[1];
                random = new Random();
            }

            // SLCO expression wrapper | x = 0.
            private boolean t_P_0_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                if(x == 0) {
                    lock_ids[0] = target_locks[0]; // Release x
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:0) | P -> P | x = 0.
            private boolean execute_transition_P_0() {
                // SLCO expression | x = 0.
                if(!(t_P_0_s_0_n_0())) {
                    return false;
                }

                currentState = CounterDistributorExact_DistributorThread.States.P;
                return true;
            }

            // SLCO expression wrapper | x = 1.
            private boolean t_P_1_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                if(x == 1) {
                    lock_ids[0] = target_locks[0]; // Release x
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:1) | P -> P | x = 1.
            private boolean execute_transition_P_1() {
                // SLCO expression | x = 1.
                if(!(t_P_1_s_0_n_0())) {
                    return false;
                }

                currentState = CounterDistributorExact_DistributorThread.States.P;
                return true;
            }

            // SLCO expression wrapper | x = 2.
            private boolean t_P_2_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                if(x == 2) {
                    lock_ids[0] = target_locks[0]; // Release x
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:2) | P -> P | x = 2.
            private boolean execute_transition_P_2() {
                // SLCO expression | x = 2.
                if(!(t_P_2_s_0_n_0())) {
                    return false;
                }

                currentState = CounterDistributorExact_DistributorThread.States.P;
                return true;
            }

            // SLCO expression wrapper | x = 3.
            private boolean t_P_3_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                if(x == 3) {
                    lock_ids[0] = target_locks[0]; // Release x
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:3) | P -> P | x = 3.
            private boolean execute_transition_P_3() {
                // SLCO expression | x = 3.
                if(!(t_P_3_s_0_n_0())) {
                    return false;
                }

                currentState = CounterDistributorExact_DistributorThread.States.P;
                return true;
            }

            // SLCO expression wrapper | x = 4.
            private boolean t_P_4_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                if(x == 4) {
                    lock_ids[0] = target_locks[0]; // Release x
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:4) | P -> P | x = 4.
            private boolean execute_transition_P_4() {
                // SLCO expression | x = 4.
                if(!(t_P_4_s_0_n_0())) {
                    return false;
                }

                currentState = CounterDistributorExact_DistributorThread.States.P;
                return true;
            }

            // SLCO expression wrapper | x = 5.
            private boolean t_P_5_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                if(x == 5) {
                    lock_ids[0] = target_locks[0]; // Release x
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:5) | P -> P | x = 5.
            private boolean execute_transition_P_5() {
                // SLCO expression | x = 5.
                if(!(t_P_5_s_0_n_0())) {
                    return false;
                }

                currentState = CounterDistributorExact_DistributorThread.States.P;
                return true;
            }

            // SLCO expression wrapper | x = 6.
            private boolean t_P_6_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                if(x == 6) {
                    lock_ids[0] = target_locks[0]; // Release x
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:6) | P -> P | x = 6.
            private boolean execute_transition_P_6() {
                // SLCO expression | x = 6.
                if(!(t_P_6_s_0_n_0())) {
                    return false;
                }

                currentState = CounterDistributorExact_DistributorThread.States.P;
                return true;
            }

            // SLCO expression wrapper | x = 7.
            private boolean t_P_7_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                if(x == 7) {
                    lock_ids[0] = target_locks[0]; // Release x
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:7) | P -> P | x = 7.
            private boolean execute_transition_P_7() {
                // SLCO expression | x = 7.
                if(!(t_P_7_s_0_n_0())) {
                    return false;
                }

                currentState = CounterDistributorExact_DistributorThread.States.P;
                return true;
            }

            // SLCO expression wrapper | x = 8.
            private boolean t_P_8_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                if(x == 8) {
                    lock_ids[0] = target_locks[0]; // Release x
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:8) | P -> P | x = 8.
            private boolean execute_transition_P_8() {
                // SLCO expression | x = 8.
                if(!(t_P_8_s_0_n_0())) {
                    return false;
                }

                currentState = CounterDistributorExact_DistributorThread.States.P;
                return true;
            }

            // SLCO expression wrapper | x = 9.
            private boolean t_P_9_s_0_n_0() {
                lock_ids[0] = target_locks[0] = 0; // Acquire x
                lockManager.acquire_locks(lock_ids, 1);
                if(x == 9) {
                    lock_ids[0] = target_locks[0]; // Release x
                    lockManager.release_locks(lock_ids, 1);
                    return true;
                }
                lock_ids[0] = target_locks[0]; // Release x
                lockManager.release_locks(lock_ids, 1);
                return false;
            }

            // SLCO transition (p:0, id:9) | P -> P | x = 9.
            private boolean execute_transition_P_9() {
                // SLCO expression | x = 9.
                if(!(t_P_9_s_0_n_0())) {
                    return false;
                }

                currentState = CounterDistributorExact_DistributorThread.States.P;
                return true;
            }

            // Attempt to fire a transition starting in state P.
            private void exec_P() {
                // [N_DET.START]
                switch(random.nextInt(10)) {
                    case 0 -> {
                        // SLCO transition (p:0, id:0) | P -> P | x = 0.
                        if(execute_transition_P_0()) {
                            return;
                        }
                    }
                    case 1 -> {
                        // SLCO transition (p:0, id:1) | P -> P | x = 1.
                        if(execute_transition_P_1()) {
                            return;
                        }
                    }
                    case 2 -> {
                        // SLCO transition (p:0, id:2) | P -> P | x = 2.
                        if(execute_transition_P_2()) {
                            return;
                        }
                    }
                    case 3 -> {
                        // SLCO transition (p:0, id:3) | P -> P | x = 3.
                        if(execute_transition_P_3()) {
                            return;
                        }
                    }
                    case 4 -> {
                        // SLCO transition (p:0, id:4) | P -> P | x = 4.
                        if(execute_transition_P_4()) {
                            return;
                        }
                    }
                    case 5 -> {
                        // SLCO transition (p:0, id:5) | P -> P | x = 5.
                        if(execute_transition_P_5()) {
                            return;
                        }
                    }
                    case 6 -> {
                        // SLCO transition (p:0, id:6) | P -> P | x = 6.
                        if(execute_transition_P_6()) {
                            return;
                        }
                    }
                    case 7 -> {
                        // SLCO transition (p:0, id:7) | P -> P | x = 7.
                        if(execute_transition_P_7()) {
                            return;
                        }
                    }
                    case 8 -> {
                        // SLCO transition (p:0, id:8) | P -> P | x = 8.
                        if(execute_transition_P_8()) {
                            return;
                        }
                    }
                    case 9 -> {
                        // SLCO transition (p:0, id:9) | P -> P | x = 9.
                        if(execute_transition_P_9()) {
                            return;
                        }
                    }
                }
                // [N_DET.END]
            }

            // Main state machine loop.
            private void exec() {
                Instant time_start = Instant.now();
                while(Duration.between(time_start, Instant.now()).toSeconds() < 30) {
                    switch(currentState) {
                        case P -> exec_P();
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
            T_Counter.start();
            T_Distributor.start();
        }

        // Join all threads.
        public void joinThreads() {
            while (true) {
                try {
                    T_Counter.join();
                    T_Distributor.join();
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
        CounterDistributor model = new CounterDistributor();
        model.startThreads();
        model.joinThreads();
    }
}