// > MODEL.START (Test)

// >> CLASS.START (P)

// VerCors verification instructions for SLCO class P.
class P {
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

// >>> STATE_MACHINE.START (SM1)

// VerCors verification instructions for SLCO state machine SM1.
class P_SM1Thread {
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
        // Reference to the parent SLCO class.
        this.c = c;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\2);
    @*/
    private void locking_operation_0() {
        //@ assume Perm(c.i, 1\2);
    }

    // SLCO expression wrapper | i >= 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\2));
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
        locking_operation_0();
        return c.i >= 0;
    }

    // SLCO expression wrapper | i < 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\2);

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\2));
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        return c.i < 2;
    }

    // SLCO expression wrapper | i >= 0 and i < 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\2));
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
        if(t_SMC0_0_s_0_n_1()) {
            if(t_SMC0_0_s_0_n_2()) {
                // Short-circuit evaluation trigger.
                return true;
            }
        } else {
        }
        // Short-circuit evaluation trigger.
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\2);

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\2) ** 0 <= c.i ** c.i < 2 ** Perm(c.x[c.i], 1\2);
    @*/
    private void locking_operation_1() {
        //@ assume 0 <= c.i && c.i < 2;
        //@ assume Perm(c.x[c.i], 1\2);
    }

    // SLCO expression wrapper | x[i] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\2);
    @*/
    private boolean t_SMC0_0_s_0_n_5() {
        locking_operation_1();
        return c.x[c.i] == 0;
    }

    // SLCO expression wrapper | i >= 0 and i < 2 and x[i] = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    private boolean t_SMC0_0_s_0_n_6() {
        if(t_SMC0_0_s_0_n_3()) {
            if(t_SMC0_0_s_0_n_5()) {
                // Short-circuit evaluation trigger.
                return true;
            }
        } else {
        }
        // Short-circuit evaluation trigger.
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0.
    private boolean execute_transition_SMC0_0() {
        // SLCO expression | i >= 0 and i < 2 and x[i] = 0.
        if(!(t_SMC0_0_s_0_n_6())) {
            return false;
        }

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    // Attempt to fire a transition starting in state SMC0.
    private void exec_SMC0() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0.
        if(execute_transition_SMC0_0()) {
            return;
        }
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;
    @*/
    // Attempt to fire a transition starting in state SMC1.
    private void exec_SMC1() {
        // There are no transitions starting in state SMC1.
    }
}

// <<< STATE_MACHINE.END (SM1)

// >>> STATE_MACHINE.START (SM2)

// VerCors verification instructions for SLCO state machine SM2.
class P_SM2Thread {
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
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        y = new int[] { 0, 0 };
        j = 1;
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

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\2);
    @*/
    private void locking_operation_2() {
        //@ assume Perm(c.i, 1\2);
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

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\2));
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
        locking_operation_2();
        return c.i >= 0;
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\2);

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\2));
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        return c.i < 2;
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

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\2));
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
        if(t_SMC0_0_s_0_n_1()) {
            if(t_SMC0_0_s_0_n_2()) {
                // Short-circuit evaluation trigger.
                return true;
            }
        } else {
        }
        // Short-circuit evaluation trigger.
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\2);

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\2) ** 0 <= c.i ** c.i < 2 ** Perm(c.x[c.i], 1\2);
    @*/
    private void locking_operation_3() {
        //@ assume 0 <= c.i && c.i < 2;
        //@ assume Perm(c.x[c.i], 1\2);
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\2);

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\2) ** 0 <= c.i ** c.i < 2 ** Perm(c.x[c.i], 1\2));
    @*/
    private boolean t_SMC0_0_s_0_n_5() {
        locking_operation_3();
        return c.x[c.i] != 0;
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

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\2) ** 0 <= c.i ** c.i < 2 ** Perm(c.x[c.i], 1\2));
    @*/
    private boolean t_SMC0_0_s_0_n_6() {
        if(t_SMC0_0_s_0_n_3()) {
            if(t_SMC0_0_s_0_n_5()) {
                // Short-circuit evaluation trigger.
                return true;
            }
        } else {
        }
        // Short-circuit evaluation trigger.
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\2) ** 0 <= c.i ** c.i < 2 ** Perm(c.x[c.i], 1\2);

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\2);
    @*/
    private void locking_operation_4() {
        // Release c.x[c.i].
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\2) ** 0 <= c.i ** c.i < 2 ** Perm(c.x[c.i], 1\2);

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\2);
    @*/
    private void assignment_0() {
        // SLCO assignment | x[i] := y[i].
        //@ assume 0 <= c.i && c.i < 2;
        //@ assert Perm(c.x[c.i], 1\2);
        //@ assume Perm(c.x[c.i], 1);
        c.x[c.i] = y[c.i];
        locking_operation_4();
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\2);
    @*/
    private void locking_operation_5() {
        // Release c.i.
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\2);
    @*/
    private void assignment_1() {
        // SLCO assignment | y[i] := 0.
        //@ assume 0 <= c.i && c.i < 2;
        //@ assert Perm(y[c.i], 1\2);
        //@ assume Perm(y[c.i], 1);
        y[c.i] = 0;
        locking_operation_5();
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
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
    private boolean execute_transition_SMC0_0() {
        // SLCO composite | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
        // SLCO expression | i >= 0 and i < 2 and x[i] != 0.
        if(!(t_SMC0_0_s_0_n_6())) {
            return false;
        }
        assignment_0();
        assignment_1();

        return true;
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
    @*/
    // Attempt to fire a transition starting in state SMC0.
    private void exec_SMC0() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
        if(execute_transition_SMC0_0()) {
            return;
        }
        // [SEQ.END]
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
    @*/
    // Attempt to fire a transition starting in state SMC1.
    private void exec_SMC1() {
        // There are no transitions starting in state SMC1.
    }
}

// <<< STATE_MACHINE.END (SM2)

// >>> STATE_MACHINE.START (SM3)

// VerCors verification instructions for SLCO state machine SM3.
class P_SM3Thread {
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
    P_SM3Thread(P c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        y = new int[] { 0, 0 };
        j = 1;
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

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\3);
    @*/
    private void locking_operation_6() {
        //@ assume Perm(c.i, 1\3);
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

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\3));
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
        locking_operation_6();
        return c.i >= 0;
    }

    // SLCO expression wrapper | i < 1.
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\3);

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\3));
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        return c.i < 1;
    }

    // SLCO expression wrapper | i >= 0 and i < 1.
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

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\3));
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
        if(t_SMC0_0_s_0_n_1()) {
            if(t_SMC0_0_s_0_n_2()) {
                // Short-circuit evaluation trigger.
                return true;
            }
        } else {
        }
        // Short-circuit evaluation trigger.
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\3);

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\3) ** 0 <= c.i ** c.i < 2 ** Perm(c.x[c.i], 1\3) ** 0 <= (c.i + 1) ** (c.i + 1) < 2 ** Perm(c.x[(c.i + 1)], 1\3);
    @*/
    private void locking_operation_7() {
        //@ assume 0 <= c.i && c.i < 2;
        //@ assume Perm(c.x[c.i], 1\3);
        //@ assume 0 <= (c.i + 1) && (c.i + 1) < 2;
        //@ assume Perm(c.x[(c.i + 1)], 1\3);
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\3);

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\3) ** 0 <= (c.i + 1) ** (c.i + 1) < 2 ** Perm(c.x[(c.i + 1)], 1\3));
    @*/
    private boolean t_SMC0_0_s_0_n_5() {
        locking_operation_7();
        return c.x[c.i] != 0;
    }

    // SLCO expression wrapper | i >= 0 and i < 1 and x[i] != 0.
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

    // Ensure that the appropriate targets are locked in the success exit branch.
    ensures \result ==> (Perm(c.i, 1\3) ** 0 <= (c.i + 1) ** (c.i + 1) < 2 ** Perm(c.x[(c.i + 1)], 1\3));
    @*/
    private boolean t_SMC0_0_s_0_n_6() {
        if(t_SMC0_0_s_0_n_3()) {
            if(t_SMC0_0_s_0_n_5()) {
                // Short-circuit evaluation trigger.
                return true;
            }
        } else {
        }
        // Short-circuit evaluation trigger.
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\3) ** 0 <= (c.i + 1) ** (c.i + 1) < 2 ** Perm(c.x[(c.i + 1)], 1\3);

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\3) ** 0 <= (c.i + 1) ** (c.i + 1) < 2 ** Perm(c.x[(c.i + 1)], 1\3);
    @*/
    private void assignment_2() {
        // SLCO assignment | i := i + 1.
        //@ assert Perm(c.i, 1\3);
        //@ assume Perm(c.i, 1);
        c.i = c.i + 1;
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\3) ** 0 <= (c.i + 1) ** (c.i + 1) < 2 ** Perm(c.x[(c.i + 1)], 1\3);

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\3);
    @*/
    private void locking_operation_8() {
        // Release c.x[(c.i + 1)].
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\3) ** 0 <= (c.i + 1) ** (c.i + 1) < 2 ** Perm(c.x[(c.i + 1)], 1\3);

    // Ensure that the appropriate targets are locked upon exit.
    ensures Perm(c.i, 1\3);
    @*/
    private void assignment_3() {
        // SLCO assignment | x[i] := y[i].
        //@ assume 0 <= c.i && c.i < 2;
        //@ assert Perm(c.x[c.i], 1\3);
        //@ assume Perm(c.x[c.i], 1);
        c.x[c.i] = y[c.i];
        locking_operation_8();
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\3);
    @*/
    private void locking_operation_9() {
        // Release c.i.
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

    // Require that the appropriate targets are locked upon entry.
    requires Perm(c.i, 1\3);
    @*/
    private void assignment_4() {
        // SLCO assignment | y[i] := 0.
        //@ assume 0 <= c.i && c.i < 2;
        //@ assert Perm(y[c.i], 1\3);
        //@ assume Perm(y[c.i], 1);
        y[c.i] = 0;
        locking_operation_9();
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
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 1 and x[i] != 0; i := i + 1; x[i] := y[i]; y[i] := 0].
    private boolean execute_transition_SMC0_0() {
        // SLCO composite | [i >= 0 and i < 1 and x[i] != 0; i := i + 1; x[i] := y[i]; y[i] := 0].
        // SLCO expression | i >= 0 and i < 1 and x[i] != 0.
        if(!(t_SMC0_0_s_0_n_6())) {
            return false;
        }
        assignment_2();
        assignment_3();
        assignment_4();

        return true;
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
    @*/
    // Attempt to fire a transition starting in state SMC0.
    private void exec_SMC0() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 1 and x[i] != 0; i := i + 1; x[i] := y[i]; y[i] := 0].
        if(execute_transition_SMC0_0()) {
            return;
        }
        // [SEQ.END]
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
    @*/
    // Attempt to fire a transition starting in state SMC1.
    private void exec_SMC1() {
        // There are no transitions starting in state SMC1.
    }
}

// <<< STATE_MACHINE.END (SM3)

// << CLASS.END (P)

// < MODEL.END (Test)