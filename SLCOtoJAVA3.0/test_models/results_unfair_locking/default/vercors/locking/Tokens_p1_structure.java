// > MODEL.START (Tokens)

// >> CLASS.START (P)

// VerCors verification instructions for SLCO class P.
class P {
    // Class variables.
    private final boolean[] tokens;
    private volatile int a;
    private volatile int b;
    private volatile int c;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.tokens, 1);
    ensures Perm(this.a, 1);
    ensures Perm(this.b, 1);
    ensures Perm(this.c, 1);

    // Require that the given values are not null.
    requires tokens != null;

    // Ensure that the right values are assigned.
    ensures this.tokens == tokens;
    ensures this.a == a;
    ensures this.b == b;
    ensures this.c == c;
    @*/
    P(boolean[] tokens, int a, int b, int c) {
        // Instantiate the class variables.
        this.tokens = tokens;
        this.a = a;
        this.b = b;
        this.c = c;
    }
}

// >>> STATE_MACHINE.START (A)

// VerCors verification instructions for SLCO state machine A.
class P_AThread {
    // The class the state machine is a part of.
    private final P c;

    // Thread local variables.
    private int x;
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
    P_AThread(P c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        x = 1;
        // Instantiate the lock requests array.
        lock_requests = new int[2];
    }

    // SLCO expression wrapper | tokens[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tokens[0]);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[0]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.tokens[0]
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.tokens[0].
        return c.tokens[0];
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (false);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[0]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_0_s_1_n_1() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.tokens[0]);

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

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | act -> act | [tokens[0]; tokens[0] := false].
    private boolean execute_transition_act_0() {
        // SLCO composite | [tokens[0]; tokens[0] := false].
        // SLCO expression | tokens[0].
        if(!(t_act_0_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | tokens[0] := false.
        //@ assert lock_requests[0] == 1; // Check c.tokens[0].
        range_check_assumption_t_0_s_2();
        c.tokens[0] = t_act_0_s_1_n_1();
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[0]
        //@ assert lock_requests[0] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | !tokens[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[0]));

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[0]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_1_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.tokens[0].
        return !(c.tokens[0]);
    }

    // SLCO expression wrapper | x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (Math.floorMod(x, 10) != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_1_s_0_n_1() {
        if(Math.floorMod(x, 10) != 0) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[0]
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            return true;
        }
        return false;
    }

    // SLCO expression wrapper | !tokens[0] and x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[0]) && Math.floorMod(x, 10) != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_1_s_0_n_2() {
        if(t_act_1_s_0_n_0()) {
            if(t_act_1_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!(c.tokens[0]) && Math.floorMod(x, 10) != 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:1, id:1) | act -> update | !tokens[0] and x % 10 != 0.
    private boolean execute_transition_act_1() {
        // SLCO expression | !tokens[0] and x % 10 != 0.
        if(!(t_act_1_s_0_n_2())) {
            return false;
        }

        return true;
    }

    // SLCO expression wrapper | !tokens[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[0]));

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_act_2_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.tokens[0].
        if(!(c.tokens[0])) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[0]
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[0]
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (Math.floorMod(x, 10) == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

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
    private boolean t_act_2_s_0_n_1() {
        return Math.floorMod(x, 10) == 0;
    }

    // SLCO expression wrapper | !tokens[0] and x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[0]) && Math.floorMod(x, 10) == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_act_2_s_0_n_2() {
        if(t_act_2_s_0_n_0()) {
            if(t_act_2_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | true.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (true);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [1: tokens[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [1: tokens[1]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [1: tokens[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_2_s_1_n_3() {
        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_2_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!(c.tokens[0]) && Math.floorMod(x, 10) == 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:1, id:2) | act -> wait | [!tokens[0] and x % 10 = 0; tokens[1] := true].
    private boolean execute_transition_act_2() {
        // SLCO composite | [!tokens[0] and x % 10 = 0; tokens[1] := true].
        // SLCO expression | !tokens[0] and x % 10 = 0.
        if(!(t_act_2_s_0_n_2())) {
            return false;
        }
        // SLCO assignment | tokens[1] := true.
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.tokens[1]
        //@ assert lock_requests[1] == 1; // Verify lock activity.

        //@ assert lock_requests[1] == 1; // Check c.tokens[1].
        range_check_assumption_t_2_s_2();
        c.tokens[1] = t_act_2_s_1_n_3();
        lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[1]
        //@ assert lock_requests[1] == 0; // Verify lock activity.

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

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
    // SLCO transition (p:0, id:0) | update -> act | true | [true; x := (641 * x + 718) % 1009; a := a + 1].
    private boolean execute_transition_update_0() {
        // (Superfluous) SLCO expression | true.

        // SLCO composite | [x := (641 * x + 718) % 1009; a := a + 1] -> [true; x := (641 * x + 718) % 1009; a := a + 1].
        // (Superfluous) SLCO expression | true.
        // SLCO assignment | x := (641 * x + 718) % 1009.
        range_check_assumption_t_0_s_3();
        x = Math.floorMod((641 * x + 718), 1009);
        // SLCO assignment | a := a + 1.
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.a
        //@ assert lock_requests[0] == 1; // Verify lock activity.

        //@ assert lock_requests[0] == 1; // Check c.a.
        range_check_assumption_t_0_s_4();
        c.a = c.a + 1;
        lock_requests[0] = lock_requests[0] - 1; // Release c.a
        //@ assert lock_requests[0] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | tokens[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tokens[0]);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[0]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_wait_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.tokens[0]
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.tokens[0].
        if(c.tokens[0]) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[0]
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (false);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[0]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_wait_0_s_1_n_1() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.tokens[0]);

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
    // SLCO transition (p:0, id:0) | wait -> wait | [tokens[0]; tokens[0] := false].
    private boolean execute_transition_wait_0() {
        // SLCO composite | [tokens[0]; tokens[0] := false].
        // SLCO expression | tokens[0].
        if(!(t_wait_0_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | tokens[0] := false.
        //@ assert lock_requests[0] == 1; // Check c.tokens[0].
        range_check_assumption_t_0_s_2();
        c.tokens[0] = t_wait_0_s_1_n_1();
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[0]
        //@ assert lock_requests[0] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | !tokens[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[1]));

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

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
    private boolean t_wait_1_s_0_n_0() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.tokens[1]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[1] == 1; // Check c.tokens[1].
        if(!(c.tokens[1])) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[1]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[1]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!(c.tokens[1]));

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
    // SLCO transition (p:1, id:1) | wait -> update | !tokens[1].
    private boolean execute_transition_wait_1() {
        // SLCO expression | !tokens[1].
        if(!(t_wait_1_s_0_n_0())) {
            return false;
        }

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

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
    // Attempt to fire a transition starting in state act.
    private void exec_act() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | act -> act | [tokens[0]; tokens[0] := false].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_act_0()) {
            return;
        }
        // [DET.START]
        // SLCO transition (p:1, id:1) | act -> update | !tokens[0] and x % 10 != 0.
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_act_1()) {
            return;
        }
        // SLCO transition (p:1, id:2) | act -> wait | [!tokens[0] and x % 10 = 0; tokens[1] := true].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_act_2()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

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
    // Attempt to fire a transition starting in state update.
    private void exec_update() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | update -> act | true | [true; x := (641 * x + 718) % 1009; a := a + 1].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_update_0()) {
            return;
        }
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

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
    // Attempt to fire a transition starting in state wait.
    private void exec_wait() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | wait -> wait | [tokens[0]; tokens[0] := false].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_wait_0()) {
            return;
        }
        // SLCO transition (p:1, id:1) | wait -> update | !tokens[1].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_wait_1()) {
            return;
        }
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (A)

// >>> STATE_MACHINE.START (B)

// VerCors verification instructions for SLCO state machine B.
class P_BThread {
    // The class the state machine is a part of.
    private final P c;

    // Thread local variables.
    private int x;
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
    P_BThread(P c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        x = 42;
        // Instantiate the lock requests array.
        lock_requests = new int[2];
    }

    // SLCO expression wrapper | tokens[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tokens[1]);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[1]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.tokens[1]
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.tokens[1].
        return c.tokens[1];
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (false);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[1]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_0_s_1_n_1() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.tokens[1]);

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

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | act -> act | [tokens[1]; tokens[1] := false].
    private boolean execute_transition_act_0() {
        // SLCO composite | [tokens[1]; tokens[1] := false].
        // SLCO expression | tokens[1].
        if(!(t_act_0_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | tokens[1] := false.
        //@ assert lock_requests[0] == 1; // Check c.tokens[1].
        range_check_assumption_t_0_s_2();
        c.tokens[1] = t_act_0_s_1_n_1();
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[1]
        //@ assert lock_requests[0] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | !tokens[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[1]));

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[1]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_1_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.tokens[1].
        return !(c.tokens[1]);
    }

    // SLCO expression wrapper | x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (Math.floorMod(x, 10) != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_1_s_0_n_1() {
        if(Math.floorMod(x, 10) != 0) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[1]
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            return true;
        }
        return false;
    }

    // SLCO expression wrapper | !tokens[1] and x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[1]) && Math.floorMod(x, 10) != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_1_s_0_n_2() {
        if(t_act_1_s_0_n_0()) {
            if(t_act_1_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!(c.tokens[1]) && Math.floorMod(x, 10) != 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:1, id:1) | act -> update | !tokens[1] and x % 10 != 0.
    private boolean execute_transition_act_1() {
        // SLCO expression | !tokens[1] and x % 10 != 0.
        if(!(t_act_1_s_0_n_2())) {
            return false;
        }

        return true;
    }

    // SLCO expression wrapper | !tokens[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[1]));

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_act_2_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.tokens[1].
        if(!(c.tokens[1])) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[1]
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[1]
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (Math.floorMod(x, 10) == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

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
    private boolean t_act_2_s_0_n_1() {
        return Math.floorMod(x, 10) == 0;
    }

    // SLCO expression wrapper | !tokens[1] and x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[1]) && Math.floorMod(x, 10) == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_act_2_s_0_n_2() {
        if(t_act_2_s_0_n_0()) {
            if(t_act_2_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | true.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (true);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [1: tokens[2]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [1: tokens[2]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [1: tokens[2]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_2_s_1_n_3() {
        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_2_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!(c.tokens[1]) && Math.floorMod(x, 10) == 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:1, id:2) | act -> wait | [!tokens[1] and x % 10 = 0; tokens[2] := true].
    private boolean execute_transition_act_2() {
        // SLCO composite | [!tokens[1] and x % 10 = 0; tokens[2] := true].
        // SLCO expression | !tokens[1] and x % 10 = 0.
        if(!(t_act_2_s_0_n_2())) {
            return false;
        }
        // SLCO assignment | tokens[2] := true.
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.tokens[2]
        //@ assert lock_requests[1] == 1; // Verify lock activity.

        //@ assert lock_requests[1] == 1; // Check c.tokens[2].
        range_check_assumption_t_2_s_2();
        c.tokens[2] = t_act_2_s_1_n_3();
        lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[2]
        //@ assert lock_requests[1] == 0; // Verify lock activity.

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

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
    // SLCO transition (p:0, id:0) | update -> act | true | [true; x := (193 * x + 953) % 1009; b := b + 1].
    private boolean execute_transition_update_0() {
        // (Superfluous) SLCO expression | true.

        // SLCO composite | [x := (193 * x + 953) % 1009; b := b + 1] -> [true; x := (193 * x + 953) % 1009; b := b + 1].
        // (Superfluous) SLCO expression | true.
        // SLCO assignment | x := (193 * x + 953) % 1009.
        range_check_assumption_t_0_s_3();
        x = Math.floorMod((193 * x + 953), 1009);
        // SLCO assignment | b := b + 1.
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.b
        //@ assert lock_requests[0] == 1; // Verify lock activity.

        //@ assert lock_requests[0] == 1; // Check c.b.
        range_check_assumption_t_0_s_4();
        c.b = c.b + 1;
        lock_requests[0] = lock_requests[0] - 1; // Release c.b
        //@ assert lock_requests[0] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | tokens[1].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tokens[1]);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[1]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_wait_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.tokens[1]
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.tokens[1].
        if(c.tokens[1]) {
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[1]
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (false);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[1]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[1]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[1]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_wait_0_s_1_n_1() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 1 && 1 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.tokens[1]);

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
    // SLCO transition (p:0, id:0) | wait -> wait | [tokens[1]; tokens[1] := false].
    private boolean execute_transition_wait_0() {
        // SLCO composite | [tokens[1]; tokens[1] := false].
        // SLCO expression | tokens[1].
        if(!(t_wait_0_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | tokens[1] := false.
        //@ assert lock_requests[0] == 1; // Check c.tokens[1].
        range_check_assumption_t_0_s_2();
        c.tokens[1] = t_wait_0_s_1_n_1();
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[1]
        //@ assert lock_requests[0] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | !tokens[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[2]));

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

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
    private boolean t_wait_1_s_0_n_0() {
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.tokens[2]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[1] == 1; // Check c.tokens[2].
        if(!(c.tokens[2])) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[2]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[2]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!(c.tokens[2]));

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
    // SLCO transition (p:1, id:1) | wait -> update | !tokens[2].
    private boolean execute_transition_wait_1() {
        // SLCO expression | !tokens[2].
        if(!(t_wait_1_s_0_n_0())) {
            return false;
        }

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

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
    // Attempt to fire a transition starting in state act.
    private void exec_act() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | act -> act | [tokens[1]; tokens[1] := false].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_act_0()) {
            return;
        }
        // [DET.START]
        // SLCO transition (p:1, id:1) | act -> update | !tokens[1] and x % 10 != 0.
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_act_1()) {
            return;
        }
        // SLCO transition (p:1, id:2) | act -> wait | [!tokens[1] and x % 10 = 0; tokens[2] := true].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_act_2()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

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
    // Attempt to fire a transition starting in state update.
    private void exec_update() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | update -> act | true | [true; x := (193 * x + 953) % 1009; b := b + 1].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_update_0()) {
            return;
        }
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

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
    // Attempt to fire a transition starting in state wait.
    private void exec_wait() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | wait -> wait | [tokens[1]; tokens[1] := false].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_wait_0()) {
            return;
        }
        // SLCO transition (p:1, id:1) | wait -> update | !tokens[2].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_wait_1()) {
            return;
        }
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (B)

// >>> STATE_MACHINE.START (C)

// VerCors verification instructions for SLCO state machine C.
class P_CThread {
    // The class the state machine is a part of.
    private final P c;

    // Thread local variables.
    private int x;
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
    P_CThread(P c) {
        // Reference to the parent SLCO class.
        this.c = c;

        // Variable instantiations.
        x = 308;
        // Instantiate the lock requests array.
        lock_requests = new int[2];
    }

    // SLCO expression wrapper | tokens[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tokens[2]);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[2]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[2], 1: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.tokens[2]
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.tokens[0]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.tokens[2].
        if(c.tokens[2]) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[0]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        return false;
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (false);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[2]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[2]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[2]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_0_s_1_n_1() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.tokens[2]);

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

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[2], 1: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | act -> act | [tokens[2]; tokens[2] := false].
    private boolean execute_transition_act_0() {
        // SLCO composite | [tokens[2]; tokens[2] := false].
        // SLCO expression | tokens[2].
        if(!(t_act_0_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | tokens[2] := false.
        //@ assert lock_requests[0] == 1; // Check c.tokens[2].
        range_check_assumption_t_0_s_2();
        c.tokens[2] = t_act_0_s_1_n_1();
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[2]
        //@ assert lock_requests[0] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | !tokens[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[2]));

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[2], 1: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[2], 1: tokens[0]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[2]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_1_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.tokens[2].
        if(!(c.tokens[2])) {
            return true;
        }
        lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[0]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (Math.floorMod(x, 10) == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[2], 1: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [1: tokens[0]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[2]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_1_s_0_n_1() {
        if(Math.floorMod(x, 10) == 0) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[2]
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[0]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | !tokens[2] and x % 10 = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[2]) && Math.floorMod(x, 10) == 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[2], 1: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [1: tokens[0]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[2]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_1_s_0_n_2() {
        if(t_act_1_s_0_n_0()) {
            if(t_act_1_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    // SLCO expression wrapper | true.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (true);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [1: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [1: tokens[0]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [1: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_act_1_s_1_n_3() {
        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_1_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!(c.tokens[2]) && Math.floorMod(x, 10) == 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[2], 1: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0 || _i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[2]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:1, id:1) | act -> wait | [!tokens[2] and x % 10 = 0; tokens[0] := true].
    private boolean execute_transition_act_1() {
        // SLCO composite | [!tokens[2] and x % 10 = 0; tokens[0] := true].
        // SLCO expression | !tokens[2] and x % 10 = 0.
        if(!(t_act_1_s_0_n_2())) {
            return false;
        }
        // SLCO assignment | tokens[0] := true.
        //@ assert lock_requests[1] == 1; // Check c.tokens[0].
        range_check_assumption_t_1_s_2();
        c.tokens[0] = t_act_1_s_1_n_3();
        lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[0]
        //@ assert lock_requests[1] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | !tokens[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[2]));

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[2]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_act_2_s_0_n_0() {
        //@ assert lock_requests[0] == 1; // Check c.tokens[2].
        if(!(c.tokens[2])) {
            lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[2]
            //@ assert lock_requests[0] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[2]
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (Math.floorMod(x, 10) != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

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
    private boolean t_act_2_s_0_n_1() {
        return Math.floorMod(x, 10) != 0;
    }

    // SLCO expression wrapper | !tokens[2] and x % 10 != 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[2]) && Math.floorMod(x, 10) != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[2]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_act_2_s_0_n_2() {
        if(t_act_2_s_0_n_0()) {
            if(t_act_2_s_0_n_1()) {
                // Short-circuit fix trigger.
                return true;
            }
        }
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!(c.tokens[2]) && Math.floorMod(x, 10) != 0);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[2]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:1, id:2) | act -> update | !tokens[2] and x % 10 != 0.
    private boolean execute_transition_act_2() {
        // SLCO expression | !tokens[2] and x % 10 != 0.
        if(!(t_act_2_s_0_n_2())) {
            return false;
        }

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

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
    // SLCO transition (p:0, id:0) | update -> act | true | [true; x := (811 * x + 31) % 1009; c := c + 1].
    private boolean execute_transition_update_0() {
        // (Superfluous) SLCO expression | true.

        // SLCO composite | [x := (811 * x + 31) % 1009; c := c + 1] -> [true; x := (811 * x + 31) % 1009; c := c + 1].
        // (Superfluous) SLCO expression | true.
        // SLCO assignment | x := (811 * x + 31) % 1009.
        range_check_assumption_t_0_s_3();
        x = Math.floorMod((811 * x + 31), 1009);
        // SLCO assignment | c := c + 1.
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.c
        //@ assert lock_requests[0] == 1; // Verify lock activity.

        //@ assert lock_requests[0] == 1; // Check c.c.
        range_check_assumption_t_0_s_4();
        c.c = c.c + 1;
        lock_requests[0] = lock_requests[0] - 1; // Release c.c
        //@ assert lock_requests[0] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | tokens[2].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tokens[2]);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that no lock requests are active prior to calling the function.
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[2]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [1: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_wait_0_s_0_n_0() {
        lock_requests[0] = lock_requests[0] + 1; // Acquire c.tokens[2]
        //@ assert lock_requests[0] == 1; // Verify lock activity.
        lock_requests[1] = lock_requests[1] + 1; // Acquire c.tokens[0]
        //@ assert lock_requests[1] == 1; // Verify lock activity.
        //@ assert lock_requests[0] == 1; // Check c.tokens[2].
        if(c.tokens[2]) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[0]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[2]
        //@ assert lock_requests[0] == 0; // Verify lock activity.
        return false;
    }

    // SLCO expression wrapper | false.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (false);

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [0: tokens[2]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the success exit of the the function:
    // - [0: tokens[2]]
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [0: tokens[2]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 0) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    private boolean t_wait_0_s_1_n_1() {
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 2 && 2 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.tokens[2]);

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

    // Ensure that that the following locks are active in the failure exit of the the function:
    // - [1: tokens[0]]
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:0, id:0) | wait -> wait | [tokens[2]; tokens[2] := false].
    private boolean execute_transition_wait_0() {
        // SLCO composite | [tokens[2]; tokens[2] := false].
        // SLCO expression | tokens[2].
        if(!(t_wait_0_s_0_n_0())) {
            return false;
        }
        // SLCO assignment | tokens[2] := false.
        //@ assert lock_requests[0] == 1; // Check c.tokens[2].
        range_check_assumption_t_0_s_2();
        c.tokens[2] = t_wait_0_s_1_n_1();
        lock_requests[0] = lock_requests[0] - 1; // Release c.tokens[2]
        //@ assert lock_requests[0] == 0; // Verify lock activity.

        return true;
    }

    // SLCO expression wrapper | !tokens[0].
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!(c.tokens[0]));

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [1: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    private boolean t_wait_1_s_0_n_0() {
        //@ assert lock_requests[1] == 1; // Check c.tokens[0].
        if(!(c.tokens[0])) {
            lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[0]
            //@ assert lock_requests[1] == 0; // Verify lock activity.
            return true;
        }
        lock_requests[1] = lock_requests[1] - 1; // Release c.tokens[0]
        //@ assert lock_requests[1] == 0; // Verify lock activity.
        return false;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 3;

    // Ensure that all state machine variable values remain unchanged.
    ensures x == \old(x);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.tokens.length; c.tokens[_i] == \old(c.tokens[_i]));
    ensures c.a == \old(c.a);
    ensures c.b == \old(c.b);
    ensures c.c == \old(c.c);
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 3;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= 0 && 0 < 3;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!(c.tokens[0]));

    // Require and ensure full permission over the lock request variable.
    context Perm(lock_requests, 1);

    // Require and ensure that the lock request array is of the correct length.
    context lock_requests != null && lock_requests.length == 2;

    // Require and ensure full permission over all lock request variable indices.
    context Perm(lock_requests[*], 1);

    // Require that that the following locks are active prior to calling the function:
    // - [1: tokens[0]]
    requires (\forall* int _i; 0 <= _i && _i < lock_requests.length; (_i == 1) ? lock_requests[_i] == 1 : lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the success exit of the function.
    ensures \result ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);

    // Ensure that that no lock requests are active in the failure exit of the function.
    ensures !(\result) ==> (\forall* int _i; 0 <= _i && _i < lock_requests.length; lock_requests[_i] == 0);
    @*/
    // SLCO transition (p:1, id:1) | wait -> update | !tokens[0].
    private boolean execute_transition_wait_1() {
        // SLCO expression | !tokens[0].
        if(!(t_wait_1_s_0_n_0())) {
            return false;
        }

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

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
    // Attempt to fire a transition starting in state act.
    private void exec_act() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | act -> act | [tokens[2]; tokens[2] := false].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_act_0()) {
            return;
        }
        // [DET.START]
        // SLCO transition (p:1, id:1) | act -> wait | [!tokens[2] and x % 10 = 0; tokens[0] := true].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_act_1()) {
            return;
        }
        // SLCO transition (p:1, id:2) | act -> update | !tokens[2] and x % 10 != 0.
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_act_2()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

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
    // Attempt to fire a transition starting in state update.
    private void exec_update() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | update -> act | true | [true; x := (811 * x + 31) % 1009; c := c + 1].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_update_0()) {
            return;
        }
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure the permission of writing to all state machine variables.
    context Perm(x, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.tokens, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.tokens != null && c.tokens.length == 3;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.tokens[*], 1);
    context Perm(c.a, 1);
    context Perm(c.b, 1);
    context Perm(c.c, 1);

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
    // Attempt to fire a transition starting in state wait.
    private void exec_wait() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | wait -> wait | [tokens[2]; tokens[2] := false].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_wait_0()) {
            return;
        }
        // SLCO transition (p:1, id:1) | wait -> update | !tokens[0].
        //@ ghost range_check_assumption_t_1();
        if(execute_transition_wait_1()) {
            return;
        }
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (C)

// << CLASS.END (P)

// < MODEL.END (Tokens)