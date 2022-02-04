// VerCors verification instructions for SLCO model Test.
// > MODEL.START (SLCOModel:Test)

// VerCors verification instructions for SLCO class P.
class SlcoClassP {

    // The class variables.
    int i; // Lock id 0
    final int[] x; // Lock id 1, length 2
}

// VerCors verification instructions for SLCO state machine SM1.
class SlcoStateMachineSM1InSlcoClassP {
    // The class the state machine is a part of.
    private final SlcoClassP c;

    // A list of lock ids and target locks that can be reused
    private final int[] lock_ids;
    private final int[] target_locks;

    // >> TRANSITION.START (Transition:SMC0.P0)

    /*@
    // Require and ensure full access to the target class. Moreover, ensure that the target class remains unchanged.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0);
    @*/
    // SLCO expression wrapper | i >= 0
    private boolean t_SMC0_0_s_0_n_0() {
        if(c.i >= 0) {
            //@ assert c.i >= 0;
            return true;
        }
        //@ assert !(c.i >= 0);
        return false;
    }

    /*@
    // Require and ensure full access to the target class. Moreover, ensure that the target class remains unchanged.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0;

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i < 2);
    @*/
    // SLCO expression wrapper | i < 2
    private boolean t_SMC0_0_s_0_n_1() {
        if(c.i < 2) {
            //@ assert c.i < 2;
            return true;
        }
        //@ assert !(c.i < 2);
        return false;
    }

    /*@
    // Require and ensure full access to the target class. Moreover, ensure that the target class remains unchanged.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2);
    @*/
    // SLCO expression wrapper | i >= 0 and i < 2
    private boolean t_SMC0_0_s_0_n_2() {
        if(!(t_SMC0_0_s_0_n_0())) {
            //@ assert !(c.i >= 0 && c.i < 2);
            return false;
        }
        //@ assert c.i >= 0;
        if(!(t_SMC0_0_s_0_n_1())) {
            //@ assert !(c.i >= 0 && c.i < 2);
            return false;
        }
        //@ assert c.i < 2;
        //@ assert c.i >= 0 && c.i < 2;
        return true;
    }

    /*@
    // Require and ensure full access to the target class. Moreover, ensure that the target class remains unchanged.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0 && c.i < 2;

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.x[c.i] == 0);
    @*/
    // SLCO expression wrapper | x[i] = 0
    private boolean t_SMC0_0_s_0_n_3() {
        if(c.x[c.i] == 0) {
            //@ assert c.x[c.i] == 0;
            return true;
        }
        //@ assert !(c.x[c.i] == 0);
        return false;
    }

    /*@
    // Require and ensure full access to the target class. Moreover, ensure that the target class remains unchanged.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2 && c.x[c.i] == 0);
    @*/
    // SLCO expression wrapper | i >= 0 and i < 2 and x[i] = 0
    private boolean t_SMC0_0_s_0_n_4() {
        if(!(t_SMC0_0_s_0_n_2())) {
            //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] == 0);
            return false;
        }
        //@ assert c.i >= 0 && c.i < 2;
        if(!(t_SMC0_0_s_0_n_3())) {
            //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] == 0);
            return false;
        }
        //@ assert c.x[c.i] == 0;
        //@ assert c.i >= 0 && c.i < 2 && c.x[c.i] == 0;
        return true;
    }

    /*@
    // Require and ensure full access to the target class. Moreover, ensure that the target class remains unchanged.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0
    private boolean execute_transition_SMC0_0() {
        // SLCO expression | i >= 0 and i < 2 and x[i] = 0
        if(!(t_SMC0_0_s_0_n_4())) {
            return false;
        }

        // currentState = SM1Thread.States.SMC0;
        return true;
    }

    // << TRANSITION.END (Transition:SMC0.P0)

}

// < MODEL.END (SLCOModel:Test)