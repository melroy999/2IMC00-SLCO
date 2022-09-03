// > MODEL.START (ToadsAndFrogs)

// >> CLASS.START (GlobalClass)

// VerCors verification instructions for SLCO class GlobalClass.
class GlobalClass {
    // Class variables.
    private volatile int y;
    private volatile int tmin;
    private volatile int fmax;
    private final int[] a;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.y, 1);
    ensures Perm(this.tmin, 1);
    ensures Perm(this.fmax, 1);
    ensures Perm(this.a, 1);

    // Require that the given values are not null.
    requires a != null;

    // Ensure that the right values are assigned.
    ensures this.y == y;
    ensures this.tmin == tmin;
    ensures this.fmax == fmax;
    ensures this.a == a;
    @*/
    GlobalClass(int y, int tmin, int fmax, int[] a) {
        // Instantiate the class variables.
        this.y = y;
        this.tmin = tmin;
        this.fmax = fmax;
        this.a = a;
    }
}

// >>> STATE_MACHINE.START (toad)

// VerCors verification instructions for SLCO state machine toad.
class GlobalClass_toadThread {
    // The class the state machine is a part of.
    private final GlobalClass c;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    GlobalClass_toadThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    // SLCO expression wrapper | y > 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_0_s_0_n_0() {
        return c.y > 0;
    }

    // SLCO expression wrapper | tmin != y - 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin != c.y - 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_0_s_0_n_1() {
        return c.tmin != c.y - 1;
    }

    // SLCO expression wrapper | y > 0 and tmin != y - 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0 && c.tmin != c.y - 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_0_s_0_n_2() {
        return t_q_0_s_0_n_0() && t_q_0_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_0_s_0_n_3() {
        return c.a[c.y - 1] == 1;
    }

    // SLCO expression wrapper | y > 0 and tmin != y - 1 and a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0 && c.tmin != c.y - 1 && c.a[c.y - 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_0_s_0_n_4() {
        return t_q_0_s_0_n_2() && t_q_0_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    pure int value_q_0_a(int _i, int _index_0, int _rhs_0, int _index_2, int _rhs_2, int v_old) = (_i == _index_2) ? _rhs_2 : ((_i == _index_0) ? _rhs_0 : v_old);
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y > 0 && c.tmin != c.y - 1 && c.a[c.y - 1] == 1);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_1;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_2;
    yields int _index_2;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (c.y == _rhs_1);
    ensures !_guard ==> (c.y == \old(c.y));
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == value_q_0_a(_i, _index_0, _rhs_0, _index_2, _rhs_2, \old(c.a[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:0) | q -> q | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
    private boolean execute_transition_q_0() {
        // SLCO composite | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
        // SLCO expression | y > 0 and tmin != y - 1 and a[y - 1] = 1.
        //@ ghost _guard = c.y > 0 && c.tmin != c.y - 1 && c.a[c.y - 1] == 1;
        if(!(t_q_0_s_0_n_4())) {
            //@ assert !(c.y > 0 && c.tmin != c.y - 1 && c.a[c.y - 1] == 1);
            return false;
        }
        //@ assert c.y > 0 && c.tmin != c.y - 1 && c.a[c.y - 1] == 1;
        // SLCO assignment | a[y] := 1.
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = 1;
        //@ ghost _index_0 = c.y;
        c.a[c.y] = 1;
        //@ assert c.a[_index_0] == _rhs_0;
        // SLCO assignment | y := y - 1.
        range_check_assumption_t_0_s_3();
        //@ ghost _rhs_1 = c.y - 1;
        c.y = c.y - 1;
        //@ assert c.y == _rhs_1;
        // SLCO assignment | a[y] := 0.
        range_check_assumption_t_0_s_4();
        //@ ghost _rhs_2 = 0;
        //@ ghost _index_2 = c.y;
        c.a[c.y] = 0;
        //@ assert c.a[_index_2] == _rhs_2;

        return true;
    }

    // SLCO expression wrapper | y > 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_1_s_0_n_0() {
        return c.y > 0;
    }

    // SLCO expression wrapper | tmin = y - 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin == c.y - 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_1_s_0_n_1() {
        return c.tmin == c.y - 1;
    }

    // SLCO expression wrapper | y > 0 and tmin = y - 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0 && c.tmin == c.y - 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_1_s_0_n_2() {
        return t_q_1_s_0_n_0() && t_q_1_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_1_s_0_n_3() {
        return c.a[c.y - 1] == 1;
    }

    // SLCO expression wrapper | y > 0 and tmin = y - 1 and a[y - 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 0 && c.tmin == c.y - 1 && c.a[c.y - 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_1_s_0_n_4() {
        return t_q_1_s_0_n_2() && t_q_1_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_5() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    pure int value_q_1_a(int _i, int _index_0, int _rhs_0, int _index_3, int _rhs_3, int v_old) = (_i == _index_3) ? _rhs_3 : ((_i == _index_0) ? _rhs_0 : v_old);
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y > 0 && c.tmin == c.y - 1 && c.a[c.y - 1] == 1);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_2;
    yields int _rhs_1;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_3;
    yields int _index_3;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (c.y == _rhs_2);
    ensures !_guard ==> (c.y == \old(c.y));
    ensures _guard ==> (c.tmin == _rhs_1);
    ensures !_guard ==> (c.tmin == \old(c.tmin));
    ensures c.fmax == \old(c.fmax);
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == value_q_1_a(_i, _index_0, _rhs_0, _index_3, _rhs_3, \old(c.a[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:1) | q -> q | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
    private boolean execute_transition_q_1() {
        // SLCO composite | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
        // SLCO expression | y > 0 and tmin = y - 1 and a[y - 1] = 1.
        //@ ghost _guard = c.y > 0 && c.tmin == c.y - 1 && c.a[c.y - 1] == 1;
        if(!(t_q_1_s_0_n_4())) {
            //@ assert !(c.y > 0 && c.tmin == c.y - 1 && c.a[c.y - 1] == 1);
            return false;
        }
        //@ assert c.y > 0 && c.tmin == c.y - 1 && c.a[c.y - 1] == 1;
        // SLCO assignment | a[y] := 1.
        range_check_assumption_t_1_s_2();
        //@ ghost _rhs_0 = 1;
        //@ ghost _index_0 = c.y;
        c.a[c.y] = 1;
        //@ assert c.a[_index_0] == _rhs_0;
        // SLCO assignment | tmin := y.
        range_check_assumption_t_1_s_3();
        //@ ghost _rhs_1 = c.y;
        c.tmin = c.y;
        //@ assert c.tmin == _rhs_1;
        // SLCO assignment | y := y - 1.
        range_check_assumption_t_1_s_4();
        //@ ghost _rhs_2 = c.y - 1;
        c.y = c.y - 1;
        //@ assert c.y == _rhs_2;
        // SLCO assignment | a[y] := 0.
        range_check_assumption_t_1_s_5();
        //@ ghost _rhs_3 = 0;
        //@ ghost _index_3 = c.y;
        c.a[c.y] = 0;
        //@ assert c.a[_index_3] == _rhs_3;

        return true;
    }

    // SLCO expression wrapper | y > 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_0() {
        return c.y > 1;
    }

    // SLCO expression wrapper | tmin != y - 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin != c.y - 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_1() {
        return c.tmin != c.y - 2;
    }

    // SLCO expression wrapper | y > 1 and tmin != y - 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin != c.y - 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_2() {
        return t_q_2_s_0_n_0() && t_q_2_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_3() {
        return c.a[c.y - 2] == 1;
    }

    // SLCO expression wrapper | y > 1 and tmin != y - 2 and a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin != c.y - 2 && c.a[c.y - 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_4() {
        return t_q_2_s_0_n_2() && t_q_2_s_0_n_3();
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_5() {
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin != c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_6() {
        return t_q_2_s_0_n_4() && t_q_2_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 2 && c.y - 2 < 9;
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    pure int value_q_2_a(int _i, int _index_0, int _rhs_0, int _index_2, int _rhs_2, int v_old) = (_i == _index_2) ? _rhs_2 : ((_i == _index_0) ? _rhs_0 : v_old);
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 2 && c.y - 2 < 9;
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y > 1 && c.tmin != c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_1;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_2;
    yields int _index_2;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (c.y == _rhs_1);
    ensures !_guard ==> (c.y == \old(c.y));
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == value_q_2_a(_i, _index_0, _rhs_0, _index_2, _rhs_2, \old(c.a[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:2) | q -> q | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
    private boolean execute_transition_q_2() {
        // SLCO composite | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
        // SLCO expression | y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
        //@ ghost _guard = c.y > 1 && c.tmin != c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2;
        if(!(t_q_2_s_0_n_6())) {
            //@ assert !(c.y > 1 && c.tmin != c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2);
            return false;
        }
        //@ assert c.y > 1 && c.tmin != c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2;
        // SLCO assignment | a[y] := 1.
        range_check_assumption_t_2_s_2();
        //@ ghost _rhs_0 = 1;
        //@ ghost _index_0 = c.y;
        c.a[c.y] = 1;
        //@ assert c.a[_index_0] == _rhs_0;
        // SLCO assignment | y := y - 2.
        range_check_assumption_t_2_s_3();
        //@ ghost _rhs_1 = c.y - 2;
        c.y = c.y - 2;
        //@ assert c.y == _rhs_1;
        // SLCO assignment | a[y] := 0.
        range_check_assumption_t_2_s_4();
        //@ ghost _rhs_2 = 0;
        //@ ghost _index_2 = c.y;
        c.a[c.y] = 0;
        //@ assert c.a[_index_2] == _rhs_2;

        return true;
    }

    // SLCO expression wrapper | y > 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_0() {
        return c.y > 1;
    }

    // SLCO expression wrapper | tmin = y - 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin == c.y - 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_1() {
        return c.tmin == c.y - 2;
    }

    // SLCO expression wrapper | y > 1 and tmin = y - 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin == c.y - 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_2() {
        return t_q_3_s_0_n_0() && t_q_3_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_3() {
        return c.a[c.y - 2] == 1;
    }

    // SLCO expression wrapper | y > 1 and tmin = y - 2 and a[y - 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin == c.y - 2 && c.a[c.y - 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_4() {
        return t_q_3_s_0_n_2() && t_q_3_s_0_n_3();
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_5() {
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.tmin == c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_6() {
        return t_q_3_s_0_n_4() && t_q_3_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_5() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 2 && c.y - 2 < 9;
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    pure int value_q_3_a(int _i, int _index_0, int _rhs_0, int _index_3, int _rhs_3, int v_old) = (_i == _index_3) ? _rhs_3 : ((_i == _index_0) ? _rhs_0 : v_old);
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 2 && c.y - 2 < 9;
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y > 1 && c.tmin == c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_2;
    yields int _rhs_1;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_3;
    yields int _index_3;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (c.y == _rhs_2);
    ensures !_guard ==> (c.y == \old(c.y));
    ensures _guard ==> (c.tmin == _rhs_1);
    ensures !_guard ==> (c.tmin == \old(c.tmin));
    ensures c.fmax == \old(c.fmax);
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == value_q_3_a(_i, _index_0, _rhs_0, _index_3, _rhs_3, \old(c.a[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:3) | q -> q | [y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; tmin := y; y := y - 2; a[y] := 0].
    private boolean execute_transition_q_3() {
        // SLCO composite | [y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; tmin := y; y := y - 2; a[y] := 0].
        // SLCO expression | y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2.
        //@ ghost _guard = c.y > 1 && c.tmin == c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2;
        if(!(t_q_3_s_0_n_6())) {
            //@ assert !(c.y > 1 && c.tmin == c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2);
            return false;
        }
        //@ assert c.y > 1 && c.tmin == c.y - 2 && c.a[c.y - 2] == 1 && c.a[c.y - 1] == 2;
        // SLCO assignment | a[y] := 1.
        range_check_assumption_t_3_s_2();
        //@ ghost _rhs_0 = 1;
        //@ ghost _index_0 = c.y;
        c.a[c.y] = 1;
        //@ assert c.a[_index_0] == _rhs_0;
        // SLCO assignment | tmin := y.
        range_check_assumption_t_3_s_3();
        //@ ghost _rhs_1 = c.y;
        c.tmin = c.y;
        //@ assert c.tmin == _rhs_1;
        // SLCO assignment | y := y - 2.
        range_check_assumption_t_3_s_4();
        //@ ghost _rhs_2 = c.y - 2;
        c.y = c.y - 2;
        //@ assert c.y == _rhs_2;
        // SLCO assignment | a[y] := 0.
        range_check_assumption_t_3_s_5();
        //@ ghost _rhs_3 = 0;
        //@ ghost _index_3 = c.y;
        c.a[c.y] = 0;
        //@ assert c.a[_index_3] == _rhs_3;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);
    @*/
    // Attempt to fire a transition starting in state q.
    private void exec_q() {
        // [SEQ.START]
        // [DET.START]
        // SLCO transition (p:0, id:0) | q -> q | [y > 0 and tmin != y - 1 and a[y - 1] = 1; a[y] := 1; y := y - 1; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | q -> q | [y > 0 and tmin = y - 1 and a[y - 1] = 1; a[y] := 1; tmin := y; y := y - 1; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_1()) {
            return;
        }
        // SLCO transition (p:0, id:2) | q -> q | [y > 1 and tmin != y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; y := y - 2; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_2()) {
            return;
        }
        // SLCO transition (p:0, id:3) | q -> q | [y > 1 and tmin = y - 2 and a[y - 2] = 1 and a[y - 1] = 2; a[y] := 1; tmin := y; y := y - 2; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_3()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (toad)

// >>> STATE_MACHINE.START (frog)

// VerCors verification instructions for SLCO state machine frog.
class GlobalClass_frogThread {
    // The class the state machine is a part of.
    private final GlobalClass c;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    GlobalClass_frogThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    // SLCO expression wrapper | y < 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_0_s_0_n_0() {
        return c.y < 8;
    }

    // SLCO expression wrapper | fmax != y + 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax != c.y + 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_0_s_0_n_1() {
        return c.fmax != c.y + 1;
    }

    // SLCO expression wrapper | y < 8 and fmax != y + 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8 && c.fmax != c.y + 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_0_s_0_n_2() {
        return t_q_0_s_0_n_0() && t_q_0_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_0_s_0_n_3() {
        return c.a[c.y + 1] == 2;
    }

    // SLCO expression wrapper | y < 8 and fmax != y + 1 and a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8 && c.fmax != c.y + 1 && c.a[c.y + 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_0_s_0_n_4() {
        return t_q_0_s_0_n_2() && t_q_0_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
    }

    /*@
    pure int value_q_0_a(int _i, int _index_0, int _rhs_0, int _index_2, int _rhs_2, int v_old) = (_i == _index_2) ? _rhs_2 : ((_i == _index_0) ? _rhs_0 : v_old);
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y < 8 && c.fmax != c.y + 1 && c.a[c.y + 1] == 2);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_1;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_2;
    yields int _index_2;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (c.y == _rhs_1);
    ensures !_guard ==> (c.y == \old(c.y));
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == value_q_0_a(_i, _index_0, _rhs_0, _index_2, _rhs_2, \old(c.a[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:0) | q -> q | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
    private boolean execute_transition_q_0() {
        // SLCO composite | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
        // SLCO expression | y < 8 and fmax != y + 1 and a[y + 1] = 2.
        //@ ghost _guard = c.y < 8 && c.fmax != c.y + 1 && c.a[c.y + 1] == 2;
        if(!(t_q_0_s_0_n_4())) {
            //@ assert !(c.y < 8 && c.fmax != c.y + 1 && c.a[c.y + 1] == 2);
            return false;
        }
        //@ assert c.y < 8 && c.fmax != c.y + 1 && c.a[c.y + 1] == 2;
        // SLCO assignment | a[y] := 2.
        range_check_assumption_t_0_s_2();
        //@ ghost _rhs_0 = 2;
        //@ ghost _index_0 = c.y;
        c.a[c.y] = 2;
        //@ assert c.a[_index_0] == _rhs_0;
        // SLCO assignment | y := y + 1.
        range_check_assumption_t_0_s_3();
        //@ ghost _rhs_1 = c.y + 1;
        c.y = c.y + 1;
        //@ assert c.y == _rhs_1;
        // SLCO assignment | a[y] := 0.
        range_check_assumption_t_0_s_4();
        //@ ghost _rhs_2 = 0;
        //@ ghost _index_2 = c.y;
        c.a[c.y] = 0;
        //@ assert c.a[_index_2] == _rhs_2;

        return true;
    }

    // SLCO expression wrapper | y < 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_1_s_0_n_0() {
        return c.y < 8;
    }

    // SLCO expression wrapper | fmax = y + 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax == c.y + 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_1_s_0_n_1() {
        return c.fmax == c.y + 1;
    }

    // SLCO expression wrapper | y < 8 and fmax = y + 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8 && c.fmax == c.y + 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_1_s_0_n_2() {
        return t_q_1_s_0_n_0() && t_q_1_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_1_s_0_n_3() {
        return c.a[c.y + 1] == 2;
    }

    // SLCO expression wrapper | y < 8 and fmax = y + 1 and a[y + 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 8 && c.fmax == c.y + 1 && c.a[c.y + 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_1_s_0_n_4() {
        return t_q_1_s_0_n_2() && t_q_1_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1_s_5() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
    }

    /*@
    pure int value_q_1_a(int _i, int _index_0, int _rhs_0, int _index_3, int _rhs_3, int v_old) = (_i == _index_3) ? _rhs_3 : ((_i == _index_0) ? _rhs_0 : v_old);
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y < 8 && c.fmax == c.y + 1 && c.a[c.y + 1] == 2);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_2;
    yields int _rhs_1;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_3;
    yields int _index_3;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (c.y == _rhs_2);
    ensures !_guard ==> (c.y == \old(c.y));
    ensures c.tmin == \old(c.tmin);
    ensures _guard ==> (c.fmax == _rhs_1);
    ensures !_guard ==> (c.fmax == \old(c.fmax));
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == value_q_1_a(_i, _index_0, _rhs_0, _index_3, _rhs_3, \old(c.a[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:1) | q -> q | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
    private boolean execute_transition_q_1() {
        // SLCO composite | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
        // SLCO expression | y < 8 and fmax = y + 1 and a[y + 1] = 2.
        //@ ghost _guard = c.y < 8 && c.fmax == c.y + 1 && c.a[c.y + 1] == 2;
        if(!(t_q_1_s_0_n_4())) {
            //@ assert !(c.y < 8 && c.fmax == c.y + 1 && c.a[c.y + 1] == 2);
            return false;
        }
        //@ assert c.y < 8 && c.fmax == c.y + 1 && c.a[c.y + 1] == 2;
        // SLCO assignment | a[y] := 2.
        range_check_assumption_t_1_s_2();
        //@ ghost _rhs_0 = 2;
        //@ ghost _index_0 = c.y;
        c.a[c.y] = 2;
        //@ assert c.a[_index_0] == _rhs_0;
        // SLCO assignment | fmax := y.
        range_check_assumption_t_1_s_3();
        //@ ghost _rhs_1 = c.y;
        c.fmax = c.y;
        //@ assert c.fmax == _rhs_1;
        // SLCO assignment | y := y + 1.
        range_check_assumption_t_1_s_4();
        //@ ghost _rhs_2 = c.y + 1;
        c.y = c.y + 1;
        //@ assert c.y == _rhs_2;
        // SLCO assignment | a[y] := 0.
        range_check_assumption_t_1_s_5();
        //@ ghost _rhs_3 = 0;
        //@ ghost _index_3 = c.y;
        c.a[c.y] = 0;
        //@ assert c.a[_index_3] == _rhs_3;

        return true;
    }

    // SLCO expression wrapper | y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_0() {
        return c.y < 7;
    }

    // SLCO expression wrapper | fmax != y + 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax != c.y + 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_1() {
        return c.fmax != c.y + 2;
    }

    // SLCO expression wrapper | y < 7 and fmax != y + 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax != c.y + 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_2() {
        return t_q_2_s_0_n_0() && t_q_2_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_3() {
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y < 7 and fmax != y + 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax != c.y + 2 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_4() {
        return t_q_2_s_0_n_2() && t_q_2_s_0_n_3();
    }

    // SLCO expression wrapper | a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_5() {
        return c.a[c.y + 2] == 2;
    }

    // SLCO expression wrapper | y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax != c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_2_s_0_n_6() {
        return t_q_2_s_0_n_4() && t_q_2_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2_s_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y + 1 && c.y + 1 < 9;
    ensures 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
    }

    /*@
    pure int value_q_2_a(int _i, int _index_0, int _rhs_0, int _index_2, int _rhs_2, int v_old) = (_i == _index_2) ? _rhs_2 : ((_i == _index_0) ? _rhs_0 : v_old);
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y + 1 && c.y + 1 < 9;
    requires 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y < 7 && c.fmax != c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_1;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_2;
    yields int _index_2;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (c.y == _rhs_1);
    ensures !_guard ==> (c.y == \old(c.y));
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == value_q_2_a(_i, _index_0, _rhs_0, _index_2, _rhs_2, \old(c.a[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:2) | q -> q | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
    private boolean execute_transition_q_2() {
        // SLCO composite | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
        // SLCO expression | y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
        //@ ghost _guard = c.y < 7 && c.fmax != c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2;
        if(!(t_q_2_s_0_n_6())) {
            //@ assert !(c.y < 7 && c.fmax != c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2);
            return false;
        }
        //@ assert c.y < 7 && c.fmax != c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2;
        // SLCO assignment | a[y] := 2.
        range_check_assumption_t_2_s_2();
        //@ ghost _rhs_0 = 2;
        //@ ghost _index_0 = c.y;
        c.a[c.y] = 2;
        //@ assert c.a[_index_0] == _rhs_0;
        // SLCO assignment | y := y + 2.
        range_check_assumption_t_2_s_3();
        //@ ghost _rhs_1 = c.y + 2;
        c.y = c.y + 2;
        //@ assert c.y == _rhs_1;
        // SLCO assignment | a[y] := 0.
        range_check_assumption_t_2_s_4();
        //@ ghost _rhs_2 = 0;
        //@ ghost _index_2 = c.y;
        c.a[c.y] = 0;
        //@ assert c.a[_index_2] == _rhs_2;

        return true;
    }

    // SLCO expression wrapper | y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_0() {
        return c.y < 7;
    }

    // SLCO expression wrapper | fmax = y + 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax == c.y + 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_1() {
        return c.fmax == c.y + 2;
    }

    // SLCO expression wrapper | y < 7 and fmax = y + 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax == c.y + 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_2() {
        return t_q_3_s_0_n_0() && t_q_3_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_3() {
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y < 7 and fmax = y + 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax == c.y + 2 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_4() {
        return t_q_3_s_0_n_2() && t_q_3_s_0_n_3();
    }

    // SLCO expression wrapper | a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_5() {
        return c.a[c.y + 2] == 2;
    }

    // SLCO expression wrapper | y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7 && c.fmax == c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_q_3_s_0_n_6() {
        return t_q_3_s_0_n_4() && t_q_3_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y && c.y < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3_s_5() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y && c.y < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y + 1 && c.y + 1 < 9;
    ensures 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
    }

    /*@
    pure int value_q_3_a(int _i, int _index_0, int _rhs_0, int _index_3, int _rhs_3, int v_old) = (_i == _index_3) ? _rhs_3 : ((_i == _index_0) ? _rhs_0 : v_old);
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y + 1 && c.y + 1 < 9;
    requires 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y < 7 && c.fmax == c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_2;
    yields int _rhs_1;
    yields int _rhs_0;
    yields int _index_0;
    yields int _rhs_3;
    yields int _index_3;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (c.y == _rhs_2);
    ensures !_guard ==> (c.y == \old(c.y));
    ensures c.tmin == \old(c.tmin);
    ensures _guard ==> (c.fmax == _rhs_1);
    ensures !_guard ==> (c.fmax == \old(c.fmax));
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == value_q_3_a(_i, _index_0, _rhs_0, _index_3, _rhs_3, \old(c.a[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:3) | q -> q | [y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; fmax := y; y := y + 2; a[y] := 0].
    private boolean execute_transition_q_3() {
        // SLCO composite | [y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; fmax := y; y := y + 2; a[y] := 0].
        // SLCO expression | y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2.
        //@ ghost _guard = c.y < 7 && c.fmax == c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2;
        if(!(t_q_3_s_0_n_6())) {
            //@ assert !(c.y < 7 && c.fmax == c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2);
            return false;
        }
        //@ assert c.y < 7 && c.fmax == c.y + 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 2;
        // SLCO assignment | a[y] := 2.
        range_check_assumption_t_3_s_2();
        //@ ghost _rhs_0 = 2;
        //@ ghost _index_0 = c.y;
        c.a[c.y] = 2;
        //@ assert c.a[_index_0] == _rhs_0;
        // SLCO assignment | fmax := y.
        range_check_assumption_t_3_s_3();
        //@ ghost _rhs_1 = c.y;
        c.fmax = c.y;
        //@ assert c.fmax == _rhs_1;
        // SLCO assignment | y := y + 2.
        range_check_assumption_t_3_s_4();
        //@ ghost _rhs_2 = c.y + 2;
        c.y = c.y + 2;
        //@ assert c.y == _rhs_2;
        // SLCO assignment | a[y] := 0.
        range_check_assumption_t_3_s_5();
        //@ ghost _rhs_3 = 0;
        //@ ghost _index_3 = c.y;
        c.a[c.y] = 0;
        //@ assert c.a[_index_3] == _rhs_3;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);
    @*/
    // Attempt to fire a transition starting in state q.
    private void exec_q() {
        // [SEQ.START]
        // [DET.START]
        // SLCO transition (p:0, id:0) | q -> q | [y < 8 and fmax != y + 1 and a[y + 1] = 2; a[y] := 2; y := y + 1; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | q -> q | [y < 8 and fmax = y + 1 and a[y + 1] = 2; a[y] := 2; fmax := y; y := y + 1; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_1()) {
            return;
        }
        // SLCO transition (p:0, id:2) | q -> q | [y < 7 and fmax != y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; y := y + 2; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_2()) {
            return;
        }
        // SLCO transition (p:0, id:3) | q -> q | [y < 7 and fmax = y + 2 and a[y + 1] = 1 and a[y + 2] = 2; a[y] := 2; fmax := y; y := y + 2; a[y] := 0].
        //@ ghost range_check_assumption_t_3();
        if(execute_transition_q_3()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (frog)

// >>> STATE_MACHINE.START (control)

// VerCors verification instructions for SLCO state machine control.
class GlobalClass_controlThread {
    // The class the state machine is a part of.
    private final GlobalClass c;

    /*@
    // Ensure full access to the class members.
    ensures Perm(this.c, 1);

    // Require that the input class is a valid object.
    requires c != null;

    // Ensure that the appropriate starter values are assigned.
    ensures this.c == c;
    @*/
    GlobalClass_controlThread(GlobalClass c) {
        // Reference to the parent SLCO class.
        this.c = c;
    }

    // SLCO expression wrapper | y = 0.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 0);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_0_s_0_n_0() {
        return c.y == 0;
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_0_s_0_n_1() {
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y = 0 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 0 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_0_s_0_n_2() {
        return t_running_0_s_0_n_0() && t_running_0_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_0_s_0_n_3() {
        return c.a[c.y + 2] == 1;
    }

    // SLCO expression wrapper | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 0 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_0_s_0_n_4() {
        return t_running_0_s_0_n_2() && t_running_0_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y + 1 && c.y + 1 < 9;
    ensures 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y + 1 && c.y + 1 < 9;
    requires 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y == 0 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:0) | running -> done | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
    private boolean execute_transition_running_0() {
        // SLCO expression | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
        //@ ghost _guard = c.y == 0 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1;
        if(!(t_running_0_s_0_n_4())) {
            //@ assert !(c.y == 0 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);
            return false;
        }
        //@ assert c.y == 0 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1;

        return true;
    }

    // SLCO expression wrapper | y = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_1_s_0_n_0() {
        return c.y == 1;
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_1_s_0_n_1() {
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y = 1 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 1 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_1_s_0_n_2() {
        return t_running_1_s_0_n_0() && t_running_1_s_0_n_1();
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_1_s_0_n_3() {
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y = 1 and a[y - 1] = 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 1 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_1_s_0_n_4() {
        return t_running_1_s_0_n_2() && t_running_1_s_0_n_3();
    }

    // SLCO expression wrapper | a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_1_s_0_n_5() {
        return c.a[c.y + 2] == 1;
    }

    // SLCO expression wrapper | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;
    context 0 <= c.y + 2 && c.y + 2 < 9;
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 1 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_1_s_0_n_6() {
        return t_running_1_s_0_n_4() && t_running_1_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 1 && c.y - 1 < 9;
    ensures 0 <= c.y + 2 && c.y + 2 < 9;
    ensures 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 1 && c.y - 1 < 9;
    requires 0 <= c.y + 2 && c.y + 2 < 9;
    requires 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y == 1 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:1) | running -> done | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
    private boolean execute_transition_running_1() {
        // SLCO expression | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
        //@ ghost _guard = c.y == 1 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1;
        if(!(t_running_1_s_0_n_6())) {
            //@ assert !(c.y == 1 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);
            return false;
        }
        //@ assert c.y == 1 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1;

        return true;
    }

    // SLCO expression wrapper | y = 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_2_s_0_n_0() {
        return c.y == 7;
    }

    // SLCO expression wrapper | a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_2_s_0_n_1() {
        return c.a[c.y - 2] == 2;
    }

    // SLCO expression wrapper | y = 7 and a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 7 && c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_2_s_0_n_2() {
        return t_running_2_s_0_n_0() && t_running_2_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_2_s_0_n_3() {
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y = 7 and a[y - 2] = 2 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_2_s_0_n_4() {
        return t_running_2_s_0_n_2() && t_running_2_s_0_n_3();
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_2_s_0_n_5() {
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_2_s_0_n_6() {
        return t_running_2_s_0_n_4() && t_running_2_s_0_n_5();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 2 && c.y - 2 < 9;
    ensures 0 <= c.y + 1 && c.y + 1 < 9;
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_2() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 2 && c.y - 2 < 9;
    requires 0 <= c.y + 1 && c.y + 1 < 9;
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y == 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:2) | running -> done | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
    private boolean execute_transition_running_2() {
        // SLCO expression | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
        //@ ghost _guard = c.y == 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1;
        if(!(t_running_2_s_0_n_6())) {
            //@ assert !(c.y == 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1);
            return false;
        }
        //@ assert c.y == 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1;

        return true;
    }

    // SLCO expression wrapper | y = 8.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 8);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_3_s_0_n_0() {
        return c.y == 8;
    }

    // SLCO expression wrapper | a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_3_s_0_n_1() {
        return c.a[c.y - 2] == 2;
    }

    // SLCO expression wrapper | y = 8 and a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 8 && c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_3_s_0_n_2() {
        return t_running_3_s_0_n_0() && t_running_3_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_3_s_0_n_3() {
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y == 8 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_3_s_0_n_4() {
        return t_running_3_s_0_n_2() && t_running_3_s_0_n_3();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 2 && c.y - 2 < 9;
    ensures 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_3() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 2 && c.y - 2 < 9;
    requires 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y == 8 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:3) | running -> done | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
    private boolean execute_transition_running_3() {
        // SLCO expression | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
        //@ ghost _guard = c.y == 8 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2;
        if(!(t_running_3_s_0_n_4())) {
            //@ assert !(c.y == 8 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2);
            return false;
        }
        //@ assert c.y == 8 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2;

        return true;
    }

    // SLCO expression wrapper | y > 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_0() {
        return c.y > 1;
    }

    // SLCO expression wrapper | y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y < 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_1() {
        return c.y < 7;
    }

    // SLCO expression wrapper | y > 1 and y < 7.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.y < 7);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_2() {
        return t_running_4_s_0_n_0() && t_running_4_s_0_n_1();
    }

    // SLCO expression wrapper | a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_3() {
        return c.a[c.y - 2] == 2;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_4() {
        return t_running_4_s_0_n_2() && t_running_4_s_0_n_3();
    }

    // SLCO expression wrapper | a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_5() {
        return c.a[c.y - 1] == 2;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_6() {
        return t_running_4_s_0_n_4() && t_running_4_s_0_n_5();
    }

    // SLCO expression wrapper | a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 1 && c.y + 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_7() {
        return c.a[c.y + 1] == 1;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 2 && c.y - 2 < 9;
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y - 1 && c.y - 1 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_8() {
        return t_running_4_s_0_n_6() && t_running_4_s_0_n_7();
    }

    // SLCO expression wrapper | a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y + 2 && c.y + 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_9() {
        return c.a[c.y + 2] == 1;
    }

    // SLCO expression wrapper | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    context 0 <= c.y - 1 && c.y - 1 < 9;
    context 0 <= c.y + 2 && c.y + 2 < 9;
    context 0 <= c.y + 1 && c.y + 1 < 9;
    context 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_running_4_s_0_n_10() {
        return t_running_4_s_0_n_8() && t_running_4_s_0_n_9();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= c.y - 1 && c.y - 1 < 9;
    ensures 0 <= c.y + 2 && c.y + 2 < 9;
    ensures 0 <= c.y + 1 && c.y + 1 < 9;
    ensures 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_4() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= c.y - 1 && c.y - 1 < 9;
        //@ assume 0 <= c.y + 2 && c.y + 2 < 9;
        //@ assume 0 <= c.y + 1 && c.y + 1 < 9;
        //@ assume 0 <= c.y - 2 && c.y - 2 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    requires 0 <= c.y - 1 && c.y - 1 < 9;
    requires 0 <= c.y + 2 && c.y + 2 < 9;
    requires 0 <= c.y + 1 && c.y + 1 < 9;
    requires 0 <= c.y - 2 && c.y - 2 < 9;

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:4) | running -> done | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
    private boolean execute_transition_running_4() {
        // SLCO expression | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
        //@ ghost _guard = c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1;
        if(!(t_running_4_s_0_n_10())) {
            //@ assert !(c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1);
            return false;
        }
        //@ assert c.y > 1 && c.y < 7 && c.a[c.y - 2] == 2 && c.a[c.y - 1] == 2 && c.a[c.y + 1] == 1 && c.a[c.y + 2] == 1;

        return true;
    }

    // SLCO expression wrapper | tmin > y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin > c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_done_0_s_0_n_0() {
        return c.tmin > c.y;
    }

    // SLCO expression wrapper | fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax < c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_done_0_s_0_n_1() {
        return c.fmax < c.y;
    }

    // SLCO expression wrapper | tmin > y and fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin > c.y && c.fmax < c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_done_0_s_0_n_2() {
        return t_done_0_s_0_n_0() && t_done_0_s_0_n_1();
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(c.tmin > c.y && c.fmax < c.y);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:0) | done -> success | tmin > y and fmax < y.
    private boolean execute_transition_done_0() {
        // SLCO expression | tmin > y and fmax < y.
        //@ ghost _guard = c.tmin > c.y && c.fmax < c.y;
        if(!(t_done_0_s_0_n_2())) {
            //@ assert !(c.tmin > c.y && c.fmax < c.y);
            return false;
        }
        //@ assert c.tmin > c.y && c.fmax < c.y;

        return true;
    }

    // SLCO expression wrapper | tmin > y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin > c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_done_1_s_0_n_0() {
        return c.tmin > c.y;
    }

    // SLCO expression wrapper | fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.fmax < c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_done_1_s_0_n_1() {
        return c.fmax < c.y;
    }

    // SLCO expression wrapper | tmin > y and fmax < y.
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (c.tmin > c.y && c.fmax < c.y);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_done_1_s_0_n_2() {
        return t_done_1_s_0_n_0() && t_done_1_s_0_n_1();
    }

    // SLCO expression wrapper | !(tmin > y and fmax < y).
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == (!((c.tmin > c.y && c.fmax < c.y)));

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private boolean t_done_1_s_0_n_3() {
        return !((t_done_1_s_0_n_2()));
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_1() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(!((c.tmin > c.y && c.fmax < c.y)));

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:1) | done -> failure | !(tmin > y and fmax < y).
    private boolean execute_transition_done_1() {
        // SLCO expression | !(tmin > y and fmax < y).
        //@ ghost _guard = !((c.tmin > c.y && c.fmax < c.y));
        if(!(t_done_1_s_0_n_3())) {
            //@ assert !(!((c.tmin > c.y && c.fmax < c.y)));
            return false;
        }
        //@ assert !((c.tmin > c.y && c.fmax < c.y));

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:0) | success -> reset | true.
    private boolean execute_transition_success_0() {
        // (Superfluous) SLCO expression | true.
        //@ ghost _guard = true;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

    // Declare the support variables.
    yields boolean _guard;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:0) | failure -> reset | true.
    private boolean execute_transition_failure_0() {
        // (Superfluous) SLCO expression | true.
        //@ ghost _guard = true;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_3() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_4() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_5() {
        
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 4 && 4 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_6() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 4 && 4 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 0 && 0 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_7() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 0 && 0 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 1 && 1 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_8() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 1 && 1 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 2 && 2 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_9() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 2 && 2 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 3 && 3 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_10() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 3 && 3 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 5 && 5 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_11() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 5 && 5 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 6 && 6 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_12() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 6 && 6 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 7 && 7 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_13() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 7 && 7 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Require and ensure that all of the accessed indices are within range.
    ensures 0 <= 8 && 8 < 9;

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0_s_14() {
        // Assume that all of the accessed indices are within range.
        //@ assume 0 <= 8 && 8 < 9;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that all class variable values remain unchanged.
    ensures c.y == \old(c.y);
    ensures c.tmin == \old(c.tmin);
    ensures c.fmax == \old(c.fmax);
    ensures (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    private void range_check_assumption_t_0() {
        
    }

    /*@
    pure int value_reset_0_a(int _i, int _index_3, int _rhs_3, int _index_4, int _rhs_4, int _index_5, int _rhs_5, int _index_6, int _rhs_6, int _index_7, int _rhs_7, int _index_8, int _rhs_8, int _index_9, int _rhs_9, int _index_10, int _rhs_10, int _index_11, int _rhs_11, int v_old) = (_i == _index_11) ? _rhs_11 : ((_i == _index_10) ? _rhs_10 : ((_i == _index_9) ? _rhs_9 : ((_i == _index_8) ? _rhs_8 : ((_i == _index_7) ? _rhs_7 : ((_i == _index_6) ? _rhs_6 : ((_i == _index_5) ? _rhs_5 : ((_i == _index_4) ? _rhs_4 : ((_i == _index_3) ? _rhs_3 : v_old))))))));
    @*/
    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);

    // Ensure that the result of the function is equivalent to the target statement.
    ensures \result == \old(true);

    // Declare the support variables.
    yields boolean _guard;
    yields int _rhs_0;
    yields int _rhs_1;
    yields int _rhs_2;
    yields int _rhs_3;
    yields int _index_3;
    yields int _rhs_4;
    yields int _index_4;
    yields int _rhs_5;
    yields int _index_5;
    yields int _rhs_6;
    yields int _index_6;
    yields int _rhs_7;
    yields int _index_7;
    yields int _rhs_8;
    yields int _index_8;
    yields int _rhs_9;
    yields int _index_9;
    yields int _rhs_10;
    yields int _index_10;
    yields int _rhs_11;
    yields int _index_11;

    // Ensure that the transition's return value is equivalent to the value of the guard.
    ensures \result == _guard;

    // Ensure that the appropriate values are changed, and if so, only when the guard holds true.
    ensures _guard ==> (c.y == _rhs_0);
    ensures !_guard ==> (c.y == \old(c.y));
    ensures _guard ==> (c.tmin == _rhs_1);
    ensures !_guard ==> (c.tmin == \old(c.tmin));
    ensures _guard ==> (c.fmax == _rhs_2);
    ensures !_guard ==> (c.fmax == \old(c.fmax));
    ensures _guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == value_reset_0_a(_i, _index_3, _rhs_3, _index_4, _rhs_4, _index_5, _rhs_5, _index_6, _rhs_6, _index_7, _rhs_7, _index_8, _rhs_8, _index_9, _rhs_9, _index_10, _rhs_10, _index_11, _rhs_11, \old(c.a[_i])));
    ensures !_guard ==> (\forall* int _i; 0 <= _i && _i < c.a.length; c.a[_i] == \old(c.a[_i]));
    @*/
    // SLCO transition (p:0, id:0) | reset -> running | true | [true; y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
    private boolean execute_transition_reset_0() {
        // (Superfluous) SLCO expression | true.
        //@ ghost _guard = true;

        // SLCO composite | [y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2] -> [true; y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
        // (Superfluous) SLCO expression | true.
        //@ ghost _guard = true;
        // SLCO assignment | y := 4.
        range_check_assumption_t_0_s_3();
        //@ ghost _rhs_0 = 4;
        c.y = 4;
        //@ assert c.y == _rhs_0;
        // SLCO assignment | tmin := 0.
        range_check_assumption_t_0_s_4();
        //@ ghost _rhs_1 = 0;
        c.tmin = 0;
        //@ assert c.tmin == _rhs_1;
        // SLCO assignment | fmax := 8.
        range_check_assumption_t_0_s_5();
        //@ ghost _rhs_2 = 8;
        c.fmax = 8;
        //@ assert c.fmax == _rhs_2;
        // SLCO assignment | a[4] := 0.
        range_check_assumption_t_0_s_6();
        //@ ghost _rhs_3 = 0;
        //@ ghost _index_3 = 4;
        c.a[4] = 0;
        //@ assert c.a[_index_3] == _rhs_3;
        // SLCO assignment | a[0] := 1.
        range_check_assumption_t_0_s_7();
        //@ ghost _rhs_4 = 1;
        //@ ghost _index_4 = 0;
        c.a[0] = 1;
        //@ assert c.a[_index_4] == _rhs_4;
        // SLCO assignment | a[1] := 1.
        range_check_assumption_t_0_s_8();
        //@ ghost _rhs_5 = 1;
        //@ ghost _index_5 = 1;
        c.a[1] = 1;
        //@ assert c.a[_index_5] == _rhs_5;
        // SLCO assignment | a[2] := 1.
        range_check_assumption_t_0_s_9();
        //@ ghost _rhs_6 = 1;
        //@ ghost _index_6 = 2;
        c.a[2] = 1;
        //@ assert c.a[_index_6] == _rhs_6;
        // SLCO assignment | a[3] := 1.
        range_check_assumption_t_0_s_10();
        //@ ghost _rhs_7 = 1;
        //@ ghost _index_7 = 3;
        c.a[3] = 1;
        //@ assert c.a[_index_7] == _rhs_7;
        // SLCO assignment | a[5] := 2.
        range_check_assumption_t_0_s_11();
        //@ ghost _rhs_8 = 2;
        //@ ghost _index_8 = 5;
        c.a[5] = 2;
        //@ assert c.a[_index_8] == _rhs_8;
        // SLCO assignment | a[6] := 2.
        range_check_assumption_t_0_s_12();
        //@ ghost _rhs_9 = 2;
        //@ ghost _index_9 = 6;
        c.a[6] = 2;
        //@ assert c.a[_index_9] == _rhs_9;
        // SLCO assignment | a[7] := 2.
        range_check_assumption_t_0_s_13();
        //@ ghost _rhs_10 = 2;
        //@ ghost _index_10 = 7;
        c.a[7] = 2;
        //@ assert c.a[_index_10] == _rhs_10;
        // SLCO assignment | a[8] := 2.
        range_check_assumption_t_0_s_14();
        //@ ghost _rhs_11 = 2;
        //@ ghost _index_11 = 8;
        c.a[8] = 2;
        //@ assert c.a[_index_11] == _rhs_11;

        return true;
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);
    @*/
    // Attempt to fire a transition starting in state running.
    private void exec_running() {
        // [SEQ.START]
        // [DET.START]
        // SLCO transition (p:0, id:0) | running -> done | y = 0 and a[y + 1] = 1 and a[y + 2] = 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_running_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | running -> done | y = 1 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_running_1()) {
            return;
        }
        // SLCO transition (p:0, id:2) | running -> done | y = 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_running_2()) {
            return;
        }
        // SLCO transition (p:0, id:3) | running -> done | y = 8 and a[y - 2] = 2 and a[y - 1] = 2.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_running_3()) {
            return;
        }
        // SLCO transition (p:0, id:4) | running -> done | y > 1 and y < 7 and a[y - 2] = 2 and a[y - 1] = 2 and a[y + 1] = 1 and a[y + 2] = 1.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_running_4()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);
    @*/
    // Attempt to fire a transition starting in state done.
    private void exec_done() {
        // [SEQ.START]
        // [DET.START]
        // SLCO transition (p:0, id:0) | done -> success | tmin > y and fmax < y.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_done_0()) {
            return;
        }
        // SLCO transition (p:0, id:1) | done -> failure | !(tmin > y and fmax < y).
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_done_1()) {
            return;
        }
        // [DET.END]
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);
    @*/
    // Attempt to fire a transition starting in state success.
    private void exec_success() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | success -> reset | true.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_success_0()) {
            return;
        }
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);
    @*/
    // Attempt to fire a transition starting in state failure.
    private void exec_failure() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | failure -> reset | true.
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_failure_0()) {
            return;
        }
        // [SEQ.END]
    }

    /*@
    // Require and ensure full access to the target class.
    context Perm(c, 1);

    // Require and ensure that the state machine has full access to the array variables within the target class.
    context Perm(c.a, 1);

    // Require and ensure that the class variable arrays are not null and of the appropriate size.
    context c.a != null && c.a.length == 9;

    // Require and ensure the permission of writing to all class variables.
    context Perm(c.y, 1);
    context Perm(c.tmin, 1);
    context Perm(c.fmax, 1);
    context Perm(c.a[*], 1);
    @*/
    // Attempt to fire a transition starting in state reset.
    private void exec_reset() {
        // [SEQ.START]
        // SLCO transition (p:0, id:0) | reset -> running | true | [true; y := 4; tmin := 0; fmax := 8; a[4] := 0; a[0] := 1; a[1] := 1; a[2] := 1; a[3] := 1; a[5] := 2; a[6] := 2; a[7] := 2; a[8] := 2].
        //@ ghost range_check_assumption_t_0();
        if(execute_transition_reset_0()) {
            return;
        }
        // [SEQ.END]
    }
}

// <<< STATE_MACHINE.END (control)

// << CLASS.END (GlobalClass)

// < MODEL.END (ToadsAndFrogs)