// > MODEL.START (Test)

// Interface for SLCO classes.
interface SLCO_Class {
    void startThreads();
    void joinThreads();
}

// VerCors verification instructions for SLCO model Test.
public class Test {
    // The objects in the model.
    private final SLCO_Class[] objects;

    Test() {
        // Instantiate the objects.
        objects = new SLCO_Class[] {
            new P(
                new int[]{ 0, 0 },
                0
            )
        };
    }
}

// > CLASS.START (P)

// VerCors verification instructions for SLCO class P.
class P implements SLCO_Class {
    // Class variables.
    private final int[] x;
    private volatile int i;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.x, 1);
    ensures Perm(this.i, 1);

    // Require that the given values are not null.
    requires x != null;

    // Ensure that the right values are assigned.
    ensures this.x == x;
    ensures this.i == i;
    @*/
    P(int[] x, int i) {
        // Instantiate the class variables.
        this.x = x;
        this.i = i;
    }
}

// VerCors verification instructions for SLCO state machine SM1.
    class P_SM1Thread extends Thread {
        // The class the state machine is a part of.
        private final P c;

        /*@
        // Ensure full access to the class members.
        ensures Perm(this.c, 1);

        // Require that the input class is a valid object.
        requires c != null;

        // Ensure that the appropriate starter values are assigned.
        ensures this.c == c;
        @*/
        P_SM1Thread(P c) {
            this.c = c;
        }

        // SLCO expression wrapper | i >= 0.
        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        private boolean t_SMC0_0_s_0_n_0() {
            lock_ids[0] = target_locks[0] = 0; // Acquire c.i
            lockManager.acquire_locks(lock_ids, 1);
            lockManager.check_lock(0); // Check c.i
            if(c.i >= 0) {
                //@ assert c.i >= 0;
                //@ node_success_closing_body
                return true;
            }
            //@ assert !(c.i >= 0);
            //@ node_failure_closing_body
            return false;
        }

        // SLCO expression wrapper | i < 2.
        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        private boolean t_SMC0_0_s_0_n_1() {
            lockManager.check_lock(0); // Check c.i
            if(c.i < 2) {
                //@ assert c.i < 2;
                //@ node_success_closing_body
                return true;
            }
            //@ assert !(c.i < 2);
            //@ node_failure_closing_body
            return false;
        }

        // SLCO expression wrapper | i >= 0 and i < 2.
        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        private boolean t_SMC0_0_s_0_n_2() {
            if(t_SMC0_0_s_0_n_0()) {
                //@ assert c.i >= 0;
                if(t_SMC0_0_s_0_n_1()) {
                    //@ assert c.i < 2;
                    //@ node_success_closing_body
                    return true;
                }
                //@ assert !(c.i < 2);
            }
            //@ assert !(c.i >= 0);
            //@ node_failure_closing_body
            return false;
        }

        // SLCO expression wrapper | x[i] = 0.
        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        private boolean t_SMC0_0_s_0_n_3() {
            lock_ids[0] = target_locks[1] = 1 + c.i; // Acquire c.x[c.i]
            lockManager.acquire_locks(lock_ids, 1);
            lockManager.check_lock(0); // Check c.i
            lockManager.check_lock(1 + c.i); // Check c.x[c.i]
            if(c.x[c.i] == 0) {
                //@ assert c.x[c.i] == 0;
                //@ node_success_closing_body
                return true;
            }
            //@ assert !(c.x[c.i] == 0);
            //@ node_failure_closing_body
            return false;
        }

        // SLCO expression wrapper | i >= 0 and i < 2 and x[i] = 0.
        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        private boolean t_SMC0_0_s_0_n_4() {
            if(t_SMC0_0_s_0_n_2()) {
                //@ assert c.i >= 0 && c.i < 2;
                if(t_SMC0_0_s_0_n_3()) {
                    //@ assert c.x[c.i] == 0;
                    //@ node_success_closing_body
                    return true;
                }
                //@ assert !(c.x[c.i] == 0);
            }
            //@ assert !(c.i >= 0 && c.i < 2);
            //@ node_failure_closing_body
            return false;
        }

        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0.
        private boolean execute_transition_SMC0_0() {
            // SLCO expression | i >= 0 and i < 2 and x[i] = 0.
            if(!(t_SMC0_0_s_0_n_4())) {
                return false;
            }

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

// VerCors verification instructions for SLCO state machine SM2.
    class P_SM2Thread extends Thread {
        // The class the state machine is a part of.
        private final P c;

        // Thread local variables.
        private int[] y;
        private int j;

        /*@
        // Ensure full access to the class members.
        ensures Perm(this.c, 1);

        // Require that the input class is a valid object.
        requires c != null;

        // Ensure that the appropriate starter values are assigned.
        ensures this.c == c;
        @*/
        P_SM2Thread(P c) {
            this.c = c;

            // Variable instantiations.
            y = new int[] { 0, 0 };
            j = 1;
        }

        // SLCO expression wrapper | i >= 0.
        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to its own array variables.
        context Perm(y, 1);

        // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
        context y != null && y.length == 2;

        // Require and ensure the permission of writing to all state machine variables.
        context Perm(y[*], 1);
        context Perm(j, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        private boolean t_SMC0_0_s_0_n_0() {
            lock_ids[0] = target_locks[0] = 0; // Acquire c.i
            lockManager.acquire_locks(lock_ids, 1);
            lockManager.check_lock(0); // Check c.i
            if(c.i >= 0) {
                //@ assert c.i >= 0;
                //@ node_success_closing_body
                return true;
            }
            //@ assert !(c.i >= 0);
            //@ node_failure_closing_body
            return false;
        }

        // SLCO expression wrapper | i < 2.
        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to its own array variables.
        context Perm(y, 1);

        // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
        context y != null && y.length == 2;

        // Require and ensure the permission of writing to all state machine variables.
        context Perm(y[*], 1);
        context Perm(j, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        private boolean t_SMC0_0_s_0_n_1() {
            lockManager.check_lock(0); // Check c.i
            if(c.i < 2) {
                //@ assert c.i < 2;
                //@ node_success_closing_body
                return true;
            }
            //@ assert !(c.i < 2);
            //@ node_failure_closing_body
            return false;
        }

        // SLCO expression wrapper | i >= 0 and i < 2.
        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to its own array variables.
        context Perm(y, 1);

        // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
        context y != null && y.length == 2;

        // Require and ensure the permission of writing to all state machine variables.
        context Perm(y[*], 1);
        context Perm(j, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        private boolean t_SMC0_0_s_0_n_2() {
            if(t_SMC0_0_s_0_n_0()) {
                //@ assert c.i >= 0;
                if(t_SMC0_0_s_0_n_1()) {
                    //@ assert c.i < 2;
                    //@ node_success_closing_body
                    return true;
                }
                //@ assert !(c.i < 2);
            }
            //@ assert !(c.i >= 0);
            //@ node_failure_closing_body
            return false;
        }

        // SLCO expression wrapper | x[i] != 0.
        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to its own array variables.
        context Perm(y, 1);

        // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
        context y != null && y.length == 2;

        // Require and ensure the permission of writing to all state machine variables.
        context Perm(y[*], 1);
        context Perm(j, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        private boolean t_SMC0_0_s_0_n_3() {
            lock_ids[0] = target_locks[1] = 1 + c.i; // Acquire c.x[c.i]
            lockManager.acquire_locks(lock_ids, 1);
            lockManager.check_lock(0); // Check c.i
            lockManager.check_lock(1 + c.i); // Check c.x[c.i]
            if(c.x[c.i] != 0) {
                //@ assert c.x[c.i] != 0;
                //@ node_success_closing_body
                return true;
            }
            //@ assert !(c.x[c.i] != 0);
            //@ node_failure_closing_body
            return false;
        }

        // SLCO expression wrapper | i >= 0 and i < 2 and x[i] != 0.
        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to its own array variables.
        context Perm(y, 1);

        // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
        context y != null && y.length == 2;

        // Require and ensure the permission of writing to all state machine variables.
        context Perm(y[*], 1);
        context Perm(j, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        private boolean t_SMC0_0_s_0_n_4() {
            if(t_SMC0_0_s_0_n_2()) {
                //@ assert c.i >= 0 && c.i < 2;
                if(t_SMC0_0_s_0_n_3()) {
                    //@ assert c.x[c.i] != 0;
                    //@ node_success_closing_body
                    return true;
                }
                //@ assert !(c.x[c.i] != 0);
            }
            //@ assert !(c.i >= 0 && c.i < 2);
            //@ node_failure_closing_body
            return false;
        }

        /*@
        // Require and ensure full access to the target class.
        context Perm(c, 1);

        // Require and ensure that the state machine has full access to its own array variables.
        context Perm(y, 1);

        // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
        context y != null && y.length == 2;

        // Require and ensure the permission of writing to all state machine variables.
        context Perm(y[*], 1);
        context Perm(j, 1);

        // Require and ensure that the state machine has full access to the array variables within the target class.
        context Perm(c.x, 1);

        // Require and ensure that the class variable arrays are not null and of the appropriate size.
        context c.x != null && c.x.length == 2;

        // Require and ensure the permission of writing to all class variables.
        context Perm(c.x[*], 1);
        context Perm(c.i, 1);
        @*/
        // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
        private boolean execute_transition_SMC0_0() {
            // SLCO composite | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
            // SLCO expression | i >= 0 and i < 2 and x[i] != 0.
            if(!(t_SMC0_0_s_0_n_4())) {
                return false;
            }
            // SLCO assignment | x[i] := y[i].
            lockManager.check_lock(0); // Check c.i
            lockManager.check_lock(1 + c.i); // Check c.x[c.i]
            c.x[c.i] = y[c.i];
            lock_ids[0] = target_locks[1]; // Release c.x[c.i]
            lockManager.release_locks(lock_ids, 1);
            // SLCO assignment | y[i] := 0.
            lockManager.check_lock(0); // Check c.i
            y[c.i] = 0;
            lock_ids[0] = target_locks[0]; // Release c.i
            lockManager.release_locks(lock_ids, 1);

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

// < CLASS.END (P)

// < MODEL.END (Test)