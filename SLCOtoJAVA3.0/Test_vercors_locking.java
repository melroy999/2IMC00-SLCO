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


    // A list of lock requests. A value of 1 denotes that the given target is locked, and 0 implies no lock.
    private final int[] lock_requests;

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

        // Instantiate the lock requests array.
        lock_requests = new int[2];
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: i]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0);
    @*/
    private boolean t_SMC0_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.i
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.i.
        if(c.i >= 0) {
            //@ assert c.i >= 0;
            return true;
        }
        //@ assert !(c.i >= 0);
        lock_requests[0] = lock_requests[0] - 1; // Release c.i
        //@ assert lock_requests[0] == 0; // Verify lock activity.
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

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0;

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: i]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: i]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i < 2);
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.i.
        if(c.i < 2) {
            //@ assert c.i < 2;
            return true;
        }
        //@ assert !(c.i < 2);
        lock_requests[0] = lock_requests[0] - 1; // Release c.i
        //@ assert lock_requests[0] == 0; // Verify lock activity.
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: i]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2);
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        if(t_SMC0_0_s_0_n_0()) {
            //@ assert c.i >= 0;
            if(t_SMC0_0_s_0_n_1()) {
                //@ assert c.i < 2;
                //@ assert c.i >= 0 && c.i < 2;
                return true;
            }
            //@ assert !(c.i < 2);
        } else {
            //@ assert !(c.i >= 0);
        }
        //@ assert !(c.i >= 0 && c.i < 2);
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

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0 && c.i < 2;

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: i]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.x[c.i] == 0);
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.x[c.i]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.i.
        //@ assert lock_requests[1] == 1; // Check c.x[c.i].
        //@ assume 0 <= c.i && c.i <= 2;
        if(c.x[c.i] == 0) {
            //@ assert c.x[c.i] == 0;
            lock_requests[0] = lock_requests[0] - 1; // Release c.i
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            lock_requests[1] = lock_requests[1] - 1; // Release c.x[c.i]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        //@ assert !(c.x[c.i] == 0);
        lock_requests[0] = lock_requests[0] - 1; // Release c.i
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.x[c.i]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2 && c.x[c.i] == 0);
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
        //@ assume 0 <= c.i && c.i <= 2;
        if(t_SMC0_0_s_0_n_2()) {
            //@ assert c.i >= 0 && c.i < 2;
            if(t_SMC0_0_s_0_n_3()) {
                //@ assert c.x[c.i] == 0;
                //@ assert c.i >= 0 && c.i < 2 && c.x[c.i] == 0;
                return true;
            }
            //@ assert !(c.x[c.i] == 0);
        } else {
            //@ assert !(c.i >= 0 && c.i < 2);
        }
        //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] == 0);
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0.
    private boolean execute_transition_SMC0_0() {
        // SLCO expression | i >= 0 and i < 2 and x[i] = 0.
        if(!(t_SMC0_0_s_0_n_4())) {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
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

    // A list of lock requests. A value of 1 denotes that the given target is locked, and 0 implies no lock.
    private final int[] lock_requests;

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

        // Instantiate the lock requests array.
        lock_requests = new int[2];
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: i]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0);
    @*/
    private boolean t_SMC0_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.i
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.i.
        if(c.i >= 0) {
            //@ assert c.i >= 0;
            return true;
        }
        //@ assert !(c.i >= 0);
        lock_requests[0] = lock_requests[0] - 1; // Release c.i
        //@ assert lock_requests[0] == 0; // Verify lock activity.
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

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0;

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: i]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: i]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i < 2);
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
        //@ assert lock_requests[0] == 1; // Check c.i.
        if(c.i < 2) {
            //@ assert c.i < 2;
            return true;
        }
        //@ assert !(c.i < 2);
        lock_requests[0] = lock_requests[0] - 1; // Release c.i
        //@ assert lock_requests[0] == 0; // Verify lock activity.
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: i]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2);
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        if(t_SMC0_0_s_0_n_0()) {
            //@ assert c.i >= 0;
            if(t_SMC0_0_s_0_n_1()) {
                //@ assert c.i < 2;
                //@ assert c.i >= 0 && c.i < 2;
                return true;
            }
            //@ assert !(c.i < 2);
        } else {
            //@ assert !(c.i >= 0);
        }
        //@ assert !(c.i >= 0 && c.i < 2);
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

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0 && c.i < 2;

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: i]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: i, 1: x[i]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.x[c.i] != 0);
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.x[c.i]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.i.
        //@ assert lock_requests[1] == 1; // Check c.x[c.i].
        //@ assume 0 <= c.i && c.i <= 2;
        if(c.x[c.i] != 0) {
            //@ assert c.x[c.i] != 0;
            return true;
        }
        //@ assert !(c.x[c.i] != 0);
        lock_requests[0] = lock_requests[0] - 1; // Release c.i
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        lock_requests[1] = lock_requests[1] - 1; // Release c.x[c.i]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: i, 1: x[i]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
        //@ assume 0 <= c.i && c.i <= 2;
        if(t_SMC0_0_s_0_n_2()) {
            //@ assert c.i >= 0 && c.i < 2;
            if(t_SMC0_0_s_0_n_3()) {
                //@ assert c.x[c.i] != 0;
                //@ assert c.i >= 0 && c.i < 2 && c.x[c.i] != 0;
                return true;
            }
            //@ assert !(c.x[c.i] != 0);
        } else {
            //@ assert !(c.i >= 0 && c.i < 2);
        }
        //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
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

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
    private boolean execute_transition_SMC0_0() {
        // SLCO composite | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
        // SLCO expression | i >= 0 and i < 2 and x[i] != 0.
        if(!(t_SMC0_0_s_0_n_4())) {
            return false;
        }
        // SLCO assignment | x[i] := y[i].
        //@ assert lock_requests[0] == 1; // Check c.i.
        //@ assert lock_requests[1] == 1; // Check c.x[c.i].
        //@ assume 0 <= c.i && c.i <= 2;
        c.x[c.i] = y[c.i];
        lock_requests[1] = lock_requests[1] - 1; // Release c.x[c.i]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        // SLCO assignment | y[i] := 0.
        //@ assert lock_requests[0] == 1; // Check c.i.
        //@ assume 0 <= c.i && c.i <= 2;
        y[c.i] = 0;
        lock_requests[0] = lock_requests[0] - 1; // Release c.i
        //@ assert lock_requests[0] == 0; // Verify lock activity.

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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active when the function terminates.
    ensures (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // Attempt to fire a transition starting in state SMC1.
    private void exec_SMC1() {
        // There are no transitions starting in state SMC1.
    }
}

// <<< STATE_MACHINE.END (SM2)

// << CLASS.END (P)

// < MODEL.END (Test)