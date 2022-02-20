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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0);
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

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0;

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i < 2);
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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);

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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.x[c.i] == 0);
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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2 && c.x[c.i] == 0);
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
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
        //@ ghost _guard = (c.i >= 0 && c.i < 2 && c.x[c.i] == 0);
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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0);
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

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0;

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i < 2);
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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.x[c.i] != 0);
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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures c.i == \old(c.i);

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));
    ensures j == \old(j);

    // Ensure that the statement's result is equivalent to the associated expression.
    ensures \result == (c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
    @*/
    private boolean t_SMC0_0_s_0_n_4() {
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
        //@ ghost _guard = (c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
        if(!(t_SMC0_0_s_0_n_4())) {
            //@ assert !(c.i >= 0 && c.i < 2 && c.x[c.i] != 0);
            return false;
        }
        //@ assert c.i >= 0 && c.i < 2 && c.x[c.i] != 0;
        // SLCO assignment | x[i] := y[i].
        //@ ghost _rhs_0 = y[c.i];
        //@ ghost _index_0 = c.i;
        c.x[c.i] = y[c.i];
        //@ assert c.x[_index_0] == _rhs_0;
        // SLCO assignment | y[i] := 0.
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

// << CLASS.END (P)

// < MODEL.END (Test)