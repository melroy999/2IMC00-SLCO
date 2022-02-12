// VerCors verification instructions for SLCO model Test.
// > MODEL.START (SLCOModel:Test)

// VerCors verification instructions for SLCO class P.
class SlcoClassP {
    // The class variables.
    int i; // Lock id 0
    final int[] x; // Lock id 1, length 2

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
    SlcoClassP(int[] x, int i) {
        // Instantiate global variables
        this.x = x;
        this.i = i;
    }
}

// VerCors verification instructions for SLCO state machine SM1.
class SlcoStateMachineSM1InSlcoClassP {
    // The class the state machine is a part of.
    private final SlcoClassP c;

    // A list of lock requests. A value of 1 denotes that the associated index is locked, and 0 implies no lock.
    private final int[] lock_requests;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);
    ensures Perm(this.lock_requests, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    ensures this.lock_requests != null && this.lock_requests.length == 2;
    @*/
    SlcoStateMachineSM1InSlcoClassP(SlcoClassP c) {
        this.c = c;
        this.lock_requests = new int[2];
    }

    // >> TRANSITION.START (Transition:SMC0.P0)

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0);

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
    @*/
    // SLCO expression wrapper | i >= 0
    private boolean t_SMC0_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire i
        //@ assert lock_requests[0] == 1;
        if(c.i >= 0) {
            //@ assert c.i >= 0;
            return true;
        }
        //@ assert !(c.i >= 0);
        lock_requests[0] = lock_requests[0] - 1; // Release i
        //@ assert lock_requests[0] == 0;
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
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0;

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i < 2);

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
    @*/
    // SLCO expression wrapper | i < 2
    private boolean t_SMC0_0_s_0_n_1() {
        if(c.i < 2) {
            //@ assert c.i < 2;
            return true;
        }
        //@ assert !(c.i < 2);
        lock_requests[0] = lock_requests[0] - 1; // Release i
        //@ assert lock_requests[0] == 0;
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
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2);

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
    // Require and ensure full access to the target class.
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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.x[c.i] == 0);

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
    @*/
    // SLCO expression wrapper | x[i] = 0
    private boolean t_SMC0_0_s_0_n_3() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire x[i]
        //@ assert lock_requests[1] == 1;
        if(c.x[c.i] == 0) {
            //@ assert c.x[c.i] == 0;
            lock_requests[0] = lock_requests[0] - 1; // Release i
                //@ assert lock_requests[0] == 0;
                lock_requests[1] = lock_requests[1] - 1; // Release x[i]
                //@ assert lock_requests[1] == 0;
            return true;
        }
        //@ assert !(c.x[c.i] == 0);
        lock_requests[0] = lock_requests[0] - 1; // Release i
        //@ assert lock_requests[0] == 0;
        lock_requests[1] = lock_requests[1] - 1; // Release x[i]
        //@ assert lock_requests[1] == 0;
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
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2 && c.x[c.i] == 0);

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
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures _guard ==> (\result == true);
    ensures !_guard ==> (\result == false);

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
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0
    private boolean execute_transition_SMC0_0() {
        // SLCO expression | i >= 0 and i < 2 and x[i] = 0
        //@ ghost _guard = c.i >= 0 && c.i < 2 && c.x[c.i] == 0;
        if(!(t_SMC0_0_s_0_n_4())) {
            //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] == 0);
            return false;
        }
        //@ assert (c.i >= 0 && c.i < 2 && c.x[c.i] == 0);

        // currentState = SM1Thread.States.SMC0;
        return true;
    }

    // << TRANSITION.END (Transition:SMC0.P0)

}

// VerCors verification instructions for SLCO state machine SM2.
class SlcoStateMachineSM2InSlcoClassP {
    // The class the state machine is a part of.
    private final SlcoClassP c;

    // A list of lock requests. A value of 1 denotes that the associated index is locked, and 0 implies no lock.
    private final int[] lock_requests;

    // State machine local variables.
    final int[] y; // length 2
    int j;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);
    ensures Perm(this.lock_requests, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    ensures this.lock_requests != null && this.lock_requests.length == 2;
    @*/
    SlcoStateMachineSM2InSlcoClassP(SlcoClassP c) {
        this.c = c;
        this.lock_requests = new int[2];

        // Variable instantiations.
        this.y = new int[] { 0, 0 };
        this.j = 1;
    }

    // >> TRANSITION.START (Transition:SMC0.P0)

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(j, 1);
    context Perm(y[*], 1);

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures (j == \old(j));

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0);

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
    @*/
    // SLCO expression wrapper | i >= 0
    private boolean t_SMC0_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire i
        //@ assert lock_requests[0] == 1;
        if(c.i >= 0) {
            //@ assert c.i >= 0;
            return true;
        }
        //@ assert !(c.i >= 0);
        lock_requests[0] = lock_requests[0] - 1; // Release i
        //@ assert lock_requests[0] == 0;
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(j, 1);
    context Perm(y[*], 1);

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0;

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures (j == \old(j));

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i < 2);

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
    @*/
    // SLCO expression wrapper | i < 2
    private boolean t_SMC0_0_s_0_n_1() {
        if(c.i < 2) {
            //@ assert c.i < 2;
            return true;
        }
        //@ assert !(c.i < 2);
        lock_requests[0] = lock_requests[0] - 1; // Release i
        //@ assert lock_requests[0] == 0;
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(j, 1);
    context Perm(y[*], 1);

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures (j == \old(j));

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2);

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
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(j, 1);
    context Perm(y[*], 1);

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0 && c.i < 2;

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures (j == \old(j));

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.x[c.i] != 0);

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
    @*/
    // SLCO expression wrapper | x[i] != 0
    private boolean t_SMC0_0_s_0_n_3() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire x[i]
        //@ assert lock_requests[1] == 1;
        if(c.x[c.i] != 0) {
            //@ assert c.x[c.i] != 0;
            return true;
        }
        //@ assert !(c.x[c.i] != 0);
        lock_requests[0] = lock_requests[0] - 1; // Release i
        //@ assert lock_requests[0] == 0;
        lock_requests[1] = lock_requests[1] - 1; // Release x[i]
        //@ assert lock_requests[1] == 0;
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(j, 1);
    context Perm(y[*], 1);

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures (j == \old(j));

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2 && c.x[c.i] != 0);

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
    @*/
    // SLCO expression wrapper | i >= 0 and i < 2 and x[i] != 0
    private boolean t_SMC0_0_s_0_n_4() {
        if(!(t_SMC0_0_s_0_n_2())) {
            //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
            return false;
        }
        //@ assert c.i >= 0 && c.i < 2;
        if(!(t_SMC0_0_s_0_n_3())) {
            //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
            return false;
        }
        //@ assert c.x[c.i] != 0;
        //@ assert c.i >= 0 && c.i < 2 && c.x[c.i] != 0;
        return true;
    }

    /*@
    // Declare helper functions for the verification of value changes.
    pure int value_SMC0_0_x(int _i, int _index_0, int _rhs_0, int v_old) = (_i == _index_0) ? _rhs_0 : v_old;
    pure int value_SMC0_0_y(int _i, int _index_1, int _rhs_1, int v_old) = (_i == _index_1) ? _rhs_1 : v_old;
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.x, 1);

    // Require and ensure that the state machine has full access to its own array variables.
    context Perm(y, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.x != null && c.x.length == 2;

    // Require and ensure that the state machine variable arrays are not null and of the appropriate size.
    context y != null && y.length == 2;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.i, 1);
    context Perm(c.x[*], 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(j, 1);
    context Perm(y[*], 1);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_1;
    yields int _index_1;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures _guard ==> (\result == true);
    ensures !_guard ==> (\result == false);

    // Ensure that the values are changed only when the guard holds true.
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == value_SMC0_0_x(_i, _index_0, _rhs_0, \old(c.x[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == value_SMC0_0_y(_i, _index_1, _rhs_1, \old(y[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));

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
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0]
    private boolean execute_transition_SMC0_0() {
        // SLCO composite | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0]
        // SLCO expression | i >= 0 and i < 2 and x[i] != 0
        //@ ghost _guard = c.i >= 0 && c.i < 2 && c.x[c.i] != 0;
        if(!(t_SMC0_0_s_0_n_4())) {
            //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
            return false;
        }
        //@ assert (c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
        // SLCO assignment | x[i] := y[i]
        //@ ghost _index_0 = c.i;
        //@ ghost _rhs_0 = y[c.i];
        c.x[c.i] = y[c.i];
        //@ assert (c.x[c.i] == y[c.i]);
        lock_requests[1] = lock_requests[1] - 1; // Release x[i]
        //@ assert lock_requests[1] == 0;
        // SLCO assignment | y[i] := 0
        //@ ghost _index_1 = c.i;
        //@ ghost _rhs_1 = 0;
        y[c.i] = 0;
        //@ assert (y[c.i] == 0);
        lock_requests[0] = lock_requests[0] - 1; // Release i
        //@ assert lock_requests[0] == 0;

        // currentState = SM2Thread.States.SMC0;
        return true;
    }

    // << TRANSITION.END (Transition:SMC0.P0)

}

// < MODEL.END (SLCOModel:Test)