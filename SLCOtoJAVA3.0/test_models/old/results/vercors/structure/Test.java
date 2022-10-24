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

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_0() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i < 2);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0 && c.i < 2);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        return t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1();
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

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x[c.i] == 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0 && c.i < 2 && c.x[c.i] == 0);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
        return t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3();
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
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

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.i >= 0 && c.i < 2 && c.x[c.i] == 0);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0.
    private boolean execute_transition_SMC0_0() {
        // SLCO expression | i >= 0 and i < 2 and x[i] = 0.
        //@ ghost _guard = c.i >= 0 && c.i < 2 && c.x[c.i] == 0;
        if(!(t_SMC0_0_s_0_n_4())) {
            //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] == 0);
            return false;
        }
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
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);
    @*/
    // Attempt to fire a transition starting in state SMC0.
    private void exec_SMC0() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0.
        //@ ghost range_check_assumption_t_0();
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
    @*/
    // Attempt to fire a transition starting in state SMC1.
    private void exec_SMC1() {
        // There are no transitions starting in state SMC1.
    }
}

// <<< STATE_MACHINE.END (SM1)

// >>> STATE_MACHINE.START (SM2a)

// VerCors verification instructions for SLCO state machine SM2a.
class P_SM2aThread {
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
    P_SM2aThread(P c) {
        // Reference to the parent SLCO class.
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

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_0() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i < 2);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0 && c.i < 2);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        return t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1();
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

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x[c.i] != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0 && c.i < 2 && c.x[c.i] != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
        return t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3();
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
    }

    /*@
    pure int value_SMC0_0_x(int _i, int _index_0, int _rhs_0, int v_old) = (_i == _index_0) ? _rhs_0 : v_old;
    @*/
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

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.i >= 0 && c.i < 2 && c.x[c.i] != 0);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;
    yields int _index_0;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == value_SMC0_0_x(_i, _index_0, _rhs_0, \old(c.x[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]].
    private boolean execute_transition_SMC0_0() {
        // SLCO composite | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]].
        // SLCO expression | i >= 0 and i < 2 and x[i] != 0.
        //@ ghost _guard = c.i >= 0 && c.i < 2 && c.x[c.i] != 0;
        if(!(t_SMC0_0_s_0_n_4())) {
            //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
            return false;
        }
        //@ assert c.i >= 0 && c.i < 2 && c.x[c.i] != 0;
        // SLCO assignment | x[i] := y[i].
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = y[c.i];
        //@ ghost _index_0 = c.i;
        c.x[c.i] = y[c.i];
        //@ assert c.x[_index_0] == _rhs_0;

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
    @*/
    // Attempt to fire a transition starting in state SMC0.
    private void exec_SMC0() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]].
        //@ ghost range_check_assumption_t_0();
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
    @*/
    // Attempt to fire a transition starting in state SMC1.
    private void exec_SMC1() {
        // There are no transitions starting in state SMC1.
    }
}

// <<< STATE_MACHINE.END (SM2a)

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

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_0() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i < 2);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0 && c.i < 2);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        return t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1();
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

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x[c.i] != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0 && c.i < 2 && c.x[c.i] != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
        return t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3();
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0_s_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
    }

    /*@
    pure int value_SMC0_0_x(int _i, int _index_0, int _rhs_0, int v_old) = (_i == _index_0) ? _rhs_0 : v_old;
    pure int value_SMC0_0_y(int _i, int _index_1, int _rhs_1, int v_old) = (_i == _index_1) ? _rhs_1 : v_old;
    @*/
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

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.i >= 0 && c.i < 2 && c.x[c.i] != 0);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_1;
    yields int _index_1;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == value_SMC0_0_x(_i, _index_0, _rhs_0, \old(c.x[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == value_SMC0_0_y(_i, _index_1, _rhs_1, \old(y[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
    private boolean execute_transition_SMC0_0() {
        // SLCO composite | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
        // SLCO expression | i >= 0 and i < 2 and x[i] != 0.
        //@ ghost _guard = c.i >= 0 && c.i < 2 && c.x[c.i] != 0;
        if(!(t_SMC0_0_s_0_n_4())) {
            //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
            return false;
        }
        //@ assert c.i >= 0 && c.i < 2 && c.x[c.i] != 0;
        // SLCO assignment | x[i] := y[i].
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = y[c.i];
        //@ ghost _index_0 = c.i;
        c.x[c.i] = y[c.i];
        //@ assert c.x[_index_0] == _rhs_0;
        // SLCO assignment | y[i] := 0.
        range_check_assumption_t_0_s_3();
        //@ ghost _rhs_1 = 0;
        //@ ghost _index_1 = c.i;
        y[c.i] = 0;
        //@ assert y[_index_1] == _rhs_1;

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
    @*/
    // Attempt to fire a transition starting in state SMC0.
    private void exec_SMC0() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2 and x[i] != 0; x[i] := y[i]; y[i] := 0].
        //@ ghost range_check_assumption_t_0();
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

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_0() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i < 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0 && c.i < 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        return t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1();
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

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x[c.i] != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0 && c.i < 1 && c.x[c.i] != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
        return t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3();
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

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0_s_2() {
        
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0_s_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0_s_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
    }

    /*@
    pure int value_SMC0_0_x(int _i, int _index_1, int _rhs_1, int v_old) = (_i == _index_1) ? _rhs_1 : v_old;
    pure int value_SMC0_0_y(int _i, int _index_2, int _rhs_2, int v_old) = (_i == _index_2) ? _rhs_2 : v_old;
    @*/
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

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.i >= 0 && c.i < 1 && c.x[c.i] != 0);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_1;
    yields int _index_1;
    yields int _rhs_0;
    yields int _rhs_2;
    yields int _index_2;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == value_SMC0_0_x(_i, _index_1, _rhs_1, \old(c.x[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures _guard ==> (c.i == _rhs_0);
    ensures !_guard ==> (c.i == \old(c.i));
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == value_SMC0_0_y(_i, _index_2, _rhs_2, \old(y[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 1 and x[i] != 0; i := i + 1; x[i] := y[i]; y[i] := 0].
    private boolean execute_transition_SMC0_0() {
        // SLCO composite | [i >= 0 and i < 1 and x[i] != 0; i := i + 1; x[i] := y[i]; y[i] := 0].
        // SLCO expression | i >= 0 and i < 1 and x[i] != 0.
        //@ ghost _guard = c.i >= 0 && c.i < 1 && c.x[c.i] != 0;
        if(!(t_SMC0_0_s_0_n_4())) {
            //@ assert !(c.i >= 0 && c.i < 1 && c.x[c.i] != 0);
            return false;
        }
        //@ assert c.i >= 0 && c.i < 1 && c.x[c.i] != 0;
        // SLCO assignment | i := i + 1.
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = c.i + 1;
        c.i = c.i + 1;
        //@ assert c.i == _rhs_0;
        // SLCO assignment | x[i] := y[i].
        range_check_assumption_t_0_s_3();
        //@ ghost _rhs_1 = y[c.i];
        //@ ghost _index_1 = c.i;
        c.x[c.i] = y[c.i];
        //@ assert c.x[_index_1] == _rhs_1;
        // SLCO assignment | y[i] := 0.
        range_check_assumption_t_0_s_4();
        //@ ghost _rhs_2 = 0;
        //@ ghost _index_2 = c.i;
        y[c.i] = 0;
        //@ assert y[_index_2] == _rhs_2;

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
    @*/
    // Attempt to fire a transition starting in state SMC0.
    private void exec_SMC0() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 1 and x[i] != 0; i := i + 1; x[i] := y[i]; y[i] := 0].
        //@ ghost range_check_assumption_t_0();
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
    @*/
    // Attempt to fire a transition starting in state SMC1.
    private void exec_SMC1() {
        // There are no transitions starting in state SMC1.
    }
}

// <<< STATE_MACHINE.END (SM3)

// >>> STATE_MACHINE.START (SM4)

// VerCors verification instructions for SLCO state machine SM4.
class P_SM4Thread {
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
    P_SM4Thread(P c) {
        // Reference to the parent SLCO class.
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

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_0() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i < 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_1() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0 && c.i < 1);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_2() {
        return t_SMC0_0_s_0_n_0() && t_SMC0_0_s_0_n_1();
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

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.x[c.i] != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_3() {
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

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.x[*], 1);
    context Perm(c.i, 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.i >= 0 && c.i < 1 && c.x[c.i] != 0);

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
        return t_SMC0_0_s_0_n_2() && t_SMC0_0_s_0_n_3();
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

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0_s_2() {
        
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

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0_s_3() {
        
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0_s_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0_s_5() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
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

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.i && c.i < 2;

    // Ensure that all state machine variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that all class variable values remain unchanged.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.i && c.i < 2;
    }

    /*@
    pure int value_SMC0_0_x(int _i, int _index_2, int _rhs_2, int v_old) = (_i == _index_2) ? _rhs_2 : v_old;
    pure int value_SMC0_0_y(int _i, int _index_3, int _rhs_3, int v_old) = (_i == _index_3) ? _rhs_3 : v_old;
    @*/
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

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.i && c.i < 2;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.i >= 0 && c.i < 1 && c.x[c.i] != 0);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_2;
    yields int _index_2;
    yields int _rhs_0;
    yields int _rhs_1;
    yields int _rhs_3;
    yields int _index_3;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == value_SMC0_0_x(_i, _index_2, _rhs_2, \old(c.x[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures _guard ==> (c.i == _rhs_1);
    ensures !_guard ==> (c.i == \old(c.i));
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == value_SMC0_0_y(_i, _index_3, _rhs_3, \old(y[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 1 and x[i] != 0; i := i + 1; i := i + 1; x[i] := y[i]; y[i] := 0].
    private boolean execute_transition_SMC0_0() {
        // SLCO composite | [i >= 0 and i < 1 and x[i] != 0; i := i + 1; i := i + 1; x[i] := y[i]; y[i] := 0].
        // SLCO expression | i >= 0 and i < 1 and x[i] != 0.
        //@ ghost _guard = c.i >= 0 && c.i < 1 && c.x[c.i] != 0;
        if(!(t_SMC0_0_s_0_n_4())) {
            //@ assert !(c.i >= 0 && c.i < 1 && c.x[c.i] != 0);
            return false;
        }
        //@ assert c.i >= 0 && c.i < 1 && c.x[c.i] != 0;
        // SLCO assignment | i := i + 1.
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = c.i + 1;
        c.i = c.i + 1;
        //@ assert c.i == _rhs_0;
        // SLCO assignment | i := i + 1.
        range_check_assumption_t_0_s_3();
        //@ ghost _rhs_1 = c.i + 1;
        c.i = c.i + 1;
        //@ assert c.i == _rhs_1;
        // SLCO assignment | x[i] := y[i].
        range_check_assumption_t_0_s_4();
        //@ ghost _rhs_2 = y[c.i];
        //@ ghost _index_2 = c.i;
        c.x[c.i] = y[c.i];
        //@ assert c.x[_index_2] == _rhs_2;
        // SLCO assignment | y[i] := 0.
        range_check_assumption_t_0_s_5();
        //@ ghost _rhs_3 = 0;
        //@ ghost _index_3 = c.i;
        y[c.i] = 0;
        //@ assert y[_index_3] == _rhs_3;

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
    @*/
    // Attempt to fire a transition starting in state SMC0.
    private void exec_SMC0() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 1 and x[i] != 0; i := i + 1; i := i + 1; x[i] := y[i]; y[i] := 0].
        //@ ghost range_check_assumption_t_0();
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
    @*/
    // Attempt to fire a transition starting in state SMC1.
    private void exec_SMC1() {
        // There are no transitions starting in state SMC1.
    }
}

// <<< STATE_MACHINE.END (SM4)

// << CLASS.END (P)

// < MODEL.END (Test)