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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

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

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

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

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures _guard ==> (\result == true);
    ensures !_guard ==> (\result == false);
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | i >= 0 and i < 2 and x[i] = 0
    private boolean execute_transition_SMC0_0() {
        // SLCO expression | i >= 0 and i < 2 and x[i] = 0
        //@ ghost _guard = t_SMC0_0_s_0_n_4();
        if(!(t_SMC0_0_s_0_n_4())) {
            return false;
        }

        // currentState = SM1Thread.States.SMC0;
        return true;
    }

    // << TRANSITION.END (Transition:SMC0.P0)

}

// VerCors verification instructions for SLCO state machine SM2.
class SlcoStateMachineSM2InSlcoClassP {
    // The class the state machine is a part of.
    private final SlcoClassP c;

    // A list of lock ids and target locks that can be reused
    private final int[] lock_ids;
    private final int[] target_locks;

    // State machine local variables.
    final int[] y; // length 2

    // >> TRANSITION.START (Transition:SMC0.P0)

    /*@
    // Require and ensure full access to the target class. Moreover, ensure that the target class remains unchanged.
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
    context Perm(y[*], 1);

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));

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
    context Perm(y[*], 1);

    // Require and ensure validity of expressions that have been encountered earlier in the control flow.
    context c.i >= 0;

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));

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
    context Perm(y[*], 1);

    // Ensure that class variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < c.x.length; c.x[_i] == \old(c.x[_i]));
    ensures (c.i == \old(c.i));

    // Ensure that state machine variable values remain unchanged after calling the function.
    ensures (\forall* int _i; 0 <= _i && _i < y.length; y[_i] == \old(y[_i]));

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
    // Declare helper functions for the verification of value changes.
    pure int value_SMC0_0_x(int _i, int _index_0, int _rhs_0, int v_old) = (_i == _index_0) ? _rhs_0 : v_old;
    pure int value_SMC0_0_y(int _i, int _index_1, int _rhs_1, int v_old) = (_i == _index_1) ? _rhs_1 : v_old;
    @*/
    /*@
    // Require and ensure full access to the target class. Moreover, ensure that the target class remains unchanged.
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
    @*/
    // SLCO transition (p:0, id:0) | SMC0 -> SMC0 | [i >= 0 and i < 2; x[i] := 0; y[i] := 0]
    private boolean execute_transition_SMC0_0() {
        // SLCO composite | [i >= 0 and i < 2; x[i] := 0; y[i] := 0]
        // SLCO expression | i >= 0 and i < 2
        //@ ghost _guard = t_SMC0_0_s_0_n_2();
        if(!(t_SMC0_0_s_0_n_2())) {
            return false;
        }
        // SLCO assignment | x[i] := 0
        //@ ghost _index_0 = c.i;
        //@ ghost _rhs_0 = 0;
        c.x[c.i] = 0;
        // SLCO assignment | y[i] := 0
        //@ ghost _index_1 = c.i;
        //@ ghost _rhs_1 = 0;
        y[c.i] = 0;

        // currentState = SM2Thread.States.SMC0;
        return true;
    }

    // << TRANSITION.END (Transition:SMC0.P0)

}

// < MODEL.END (SLCOModel:Test)